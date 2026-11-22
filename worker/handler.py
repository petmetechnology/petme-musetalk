"""petmeai.com 精确口型 worker: one RunPod Serverless job = one video call (functions/petmt.js starts it).

Job input: { sid, secret, key, video, fn }.  The job:
  1. downloads the character's idle video, loads its avatar (cached in the bucket via fn cache_get / cache_put; ~1 min when new)
  2. serves a WebSocket (FastAPI, started once per worker) behind a Cloudflare quick tunnel (also once per worker) and reports
     fn ?op=ready {url}; every minute fn ?op=lease (the page's VIP credit pays per minute; {stop:true} ends the call)
  3. returns when the page hangs up (no client for GRACE s), the lease says stop, or nobody connects within FIRST_WAIT s.

WebSocket /ws?k=<key>  (one client per job)
  server -> client  text  {"type":"hello", n, fps, w, h, gpu}            then one binary 0x05 per idle frame, then {"type":"idle_done"}
                    bin   0x05 + u16 idx + u16 x + u16 y + JPEG          silent mouth patch of idle frame idx (720x1280 px)
                    bin   0x02 + u32 id + u16 k + u16 x + u16 y + JPEG   mouth patch for audio frame k (40 ms) of sentence id
                    text  {"type":"said", id, frames, fps, first_ms} | {"type":"bye", reason} | {"type":"pong"}
  client -> server  bin   0x06 + u32 id + f32 i0 + f32 step + int16 PCM 16 kHz   one sentence: audio frame k is drawn on idle
                                                                                 frame round(i0 + k*step) % n (step = video fps / 25)
                    text  {"type":"cancel"} (drop everything queued / running) | {"type":"ping"}

Local test without a GPU / RunPod:  MOCK=1 LOCAL=1 python handler.py <idle.mp4> [port]   -> ws://localhost:<port>/ws?k=local
"""
import asyncio, json, os, re, struct, subprocess, sys, tempfile, threading, time, traceback
import numpy as np
import httpx

MOCK = bool(os.environ.get('MOCK'))
LOCAL = bool(os.environ.get('LOCAL'))
PORT = int(os.environ.get('PORT', '8000'))
REPO = os.environ.get('MUSETALK_DIR', '/opt/MuseTalk')
FIRST_WAIT = 150        # s: the page must connect within this after ready
GRACE = 25              # s: page gone this long (closed tab, lost network) -> end the call
LEASE_EVERY = 60

if MOCK:
    from engine import MockModels as Models, MockAvatar as Avatar
else:
    from engine import Models, Avatar

t_boot = time.time()
M = Models() if MOCK else Models(REPO, batch=int(os.environ.get('BATCH', '8')))
gpu_lock = threading.Lock()


class Job:
    """the call this worker serves right now"""
    def __init__(self, key, avatar):
        self.key, self.avatar = key, avatar
        self.clients = 0
        self.last_seen = time.time()
        self.ever = False
        self.stop = threading.Event()
        self.reason = ''
        self.ws_close = []          # callbacks that close the open socket (bye)


CUR = None

# ---------------- web server (started once per worker) ----------------
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
app = FastAPI()


@app.get('/ping')
async def ping():
    return JSONResponse({'ok': True, 'busy': CUR is not None})


@app.websocket('/ws')
async def ws_main(ws: WebSocket):
    job = CUR
    await ws.accept()
    if job is None or ws.query_params.get('k') != job.key or job.stop.is_set():
        await ws.send_text(json.dumps({'type': 'bye', 'reason': 'key'})); await ws.close(); return
    if job.clients:
        await ws.send_text(json.dumps({'type': 'bye', 'reason': 'busy'})); await ws.close(); return
    job.clients += 1; job.ever = True; job.last_seen = time.time()
    loop = asyncio.get_running_loop()
    out = asyncio.Queue()
    put = lambda m: loop.call_soon_threadsafe(out.put_nowait, m)
    cancel = threading.Event()
    work = asyncio.Queue()
    av = job.avatar
    sent = {'bytes': 0}

    async def sender():
        while True:
            m = await out.get()
            if m is None:
                return
            if isinstance(m, (bytes, bytearray)):
                sent['bytes'] += len(m); await ws.send_bytes(m)
            else:
                await ws.send_text(json.dumps(m))

    async def worker():
        while True:
            item = await work.get()
            if item is None:
                return
            sid, i0, step, pcm = item
            if cancel.is_set():
                continue

            def emit(k, x, y, jpg, sid=sid):
                put(b'\x02' + struct.pack('<IHHH', sid, k, x, y) + jpg)

            def run():
                with gpu_lock:
                    return av.generate(pcm, lambda k: int(round(i0 + k * step)), emit, cancel)
            try:
                st = await asyncio.to_thread(run)
                st.update(type='said', id=sid); put(st)
            except Exception as e:
                traceback.print_exc(); put({'type': 'said', 'id': sid, 'error': str(e)[:200]})

    def bye(reason):
        put({'type': 'bye', 'reason': reason}); put(None)
    job.ws_close.append(bye)
    tasks = [asyncio.create_task(sender()), asyncio.create_task(worker())]
    put({'type': 'hello', 'n': av.n, 'fps': av.vfps, 'w': 720, 'h': 1280, 'gpu': M.gpu})
    for i, (x, y, jpg) in enumerate(av.silent):
        put(b'\x05' + struct.pack('<HHH', i, x, y) + jpg)
    put({'type': 'idle_done'})
    try:
        while not job.stop.is_set():
            m = await ws.receive()
            job.last_seen = time.time()
            if m.get('type') == 'websocket.disconnect':
                break
            b = m.get('bytes')
            if b:
                if b[0] == 6 and len(b) > 13:
                    sid, i0, step = struct.unpack('<Iff', b[1:13])
                    pcm = np.frombuffer(b[13:13 + (len(b) - 13) // 2 * 2], np.int16)
                    pad = (-len(pcm)) % 640
                    if pad:
                        pcm = np.concatenate([pcm, np.zeros(pad, np.int16)])
                    if len(pcm) > 16000 * 60:            # one sentence is never a minute long
                        pcm = pcm[:16000 * 60]
                    cancel.clear()
                    work.put_nowait((sid, i0, step, pcm))
                continue
            j = json.loads(m.get('text') or '{}')
            if j.get('type') == 'cancel':
                cancel.set()
                while not work.empty():
                    work.get_nowait()
            elif j.get('type') == 'ping':
                put({'type': 'pong'})
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        cancel.set(); work.put_nowait(None); put(None)
        for t in tasks:
            t.cancel()
        job.clients -= 1; job.last_seen = time.time()
        if bye in job.ws_close:
            job.ws_close.remove(bye)
        print(f'[ws] client left, sent {sent["bytes"] / 1e6:.1f} MB', flush=True)


def start_server():
    import uvicorn
    threading.Thread(target=lambda: uvicorn.run(app, host='0.0.0.0', port=PORT, log_level='warning'), daemon=True).start()
    for _ in range(100):
        try:
            if httpx.get(f'http://127.0.0.1:{PORT}/ping', timeout=1).status_code == 200:
                return
        except Exception:
            time.sleep(0.1)
    raise RuntimeError('web server did not start')


TUNNEL = {'url': None, 'proc': None}


def tunnel_url():
    """a Cloudflare quick tunnel to the web server, started once per worker (restarted if it died)"""
    p = TUNNEL['proc']
    if p and p.poll() is None and TUNNEL['url']:
        return TUNNEL['url']
    log = tempfile.NamedTemporaryFile('w+', suffix='.log', delete=False)
    p = subprocess.Popen(['cloudflared', 'tunnel', '--no-autoupdate', '--url', f'http://127.0.0.1:{PORT}'], stdout=log, stderr=subprocess.STDOUT)
    TUNNEL['proc'] = p
    for _ in range(300):
        time.sleep(0.2)
        m = re.search(r'https://[a-z0-9-]+\.trycloudflare\.com', open(log.name).read())
        if m:
            TUNNEL['url'] = m.group(0)
            break
    if not TUNNEL['url']:
        raise RuntimeError('no tunnel url')
    # the quick tunnel's DNS name needs a moment before it resolves
    for _ in range(40):
        try:
            if httpx.get(TUNNEL['url'] + '/ping', timeout=4).status_code == 200:
                break
        except Exception:
            pass
        time.sleep(0.5)
    return TUNNEL['url']


# ---------------- the job ----------------
def fn_call(inp, op, extra=None, timeout=60, **kw):
    q = {'op': op, 'sid': inp['sid'], 'secret': inp['secret'], **(extra or {})}
    return httpx.post(inp['fn'], params=q, timeout=timeout, **kw)


def handler(job):
    global CUR
    inp = job['input']
    t0 = time.time()
    try:
        d = tempfile.mkdtemp()
        vp = os.path.join(d, 'idle.mp4')
        r = httpx.get(inp['video'], timeout=60, follow_redirects=True); r.raise_for_status()
        open(vp, 'wb').write(r.content)
        cache = None
        try:
            c = fn_call(inp, 'cache_get')
            if c.status_code == 200:
                cache = c.content
        except Exception as e:
            print('[job] cache_get', e, flush=True)
        av = Avatar(M, vp, cache)
        if av.fresh:
            try:
                fn_call(inp, 'cache_put', {'name': 'avatar.bin'}, timeout=120, content=av.to_bytes(),
                        headers={'Content-Type': 'application/octet-stream'})
            except Exception as e:
                print('[job] cache_put', e, flush=True)
        J = Job(inp['key'], av)
        CUR = J
        url = tunnel_url()
        r = fn_call(inp, 'ready', json={'url': url, 'n': av.n, 'gpu': M.gpu, 'ms': int((time.time() - t0) * 1000)}).json()
        print(f'[job] ready in {time.time() - t0:.1f}s (worker up {time.time() - t_boot:.0f}s) {url}', flush=True)
        if r.get('stop'):
            J.reason = r.get('reason', 'stop'); return {'ok': False, 'reason': J.reason}
        last_lease = t_ready = time.time()
        while True:
            time.sleep(1)
            now = time.time()
            if J.clients == 0 and J.ever and now - J.last_seen > GRACE:
                J.reason = 'left'; break
            if not J.ever and now - t_ready > FIRST_WAIT:
                J.reason = 'never'; break
            if now - last_lease >= LEASE_EVERY:
                last_lease = now
                try:
                    lr = fn_call(inp, 'lease').json()
                    if lr.get('stop'):
                        J.reason = lr.get('reason', 'stop'); break
                except Exception as e:
                    print('[job] lease', e, flush=True)      # a hiccup: try again next minute
        return {'ok': True, 'reason': J.reason, 'sec': int(time.time() - t0)}
    except Exception as e:
        traceback.print_exc()
        return {'ok': False, 'error': str(e)[:300]}
    finally:
        J = CUR
        CUR = None
        if J:
            J.stop.set()
            for f in list(J.ws_close):
                try:
                    f(J.reason or 'end')
                except Exception:
                    pass
            if J.reason not in ('user', 'replaced', 'credit', 'max', 'stop', 'timeout'):
                try:
                    fn_call(inp, 'end', {'reason': J.reason or 'worker'}, timeout=20)
                except Exception:
                    pass


if __name__ == '__main__':
    start_server()
    if LOCAL:
        # one local call, no RunPod / tunnel / billing: ws://localhost:PORT/ws?k=local
        CUR = Job('local', Avatar(M, sys.argv[1]))
        print(f'[local] ws://127.0.0.1:{PORT}/ws?k=local  frames={CUR.avatar.n} fps={CUR.avatar.vfps}', flush=True)
        while True:
            time.sleep(3600)
    import runpod
    runpod.serverless.start({'handler': handler})
