"""MuseTalk 1.5 engine for the petmeai.com 精确口型 worker (RunPod Serverless).

Models are loaded once per worker process (they stay warm between jobs on the same worker). An Avatar = one character's idle
loop, every frame kept (the page's frame index = round(mediaTime * fps) % frames must address the same frames), scaled to
720x1280. Preparation (~1 min, cached as bytes by the caller): 68-point landmarks (face_alignment / FAN, the same 68-point
layout as MuseTalk's DWPose face) -> MuseTalk's face box rule -> temporally smoothed -> VAE latents of the 256x256 crop,
face-parsing blend mask ('jaw') and the patch rectangle (aligned to 16 px) -> a silent patch for every frame (the mouth
re-rendered with silence, so it looks the same while idle and while talking: no visible switch).

generate(pcm16, idxs, emit): whisper features for one sentence -> UNet + VAE decode in batches of 8 -> blend only the patch
rectangle of idle frame idxs[k] -> emit(k, x, y, jpeg).
"""
import io, os, sys, time, pickle
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2

FPS = 25
W, H = 720, 1280
JPEG_Q = 88
POOL = ThreadPoolExecutor(max(2, min(6, (os.cpu_count() or 4))))   # blend + JPEG of a batch's frames in parallel (cv2 / numpy release the GIL)
SILENT_V = 3      # 3: frames decoded as BT.709 (11-22). bump when _blend changes: cached silent patches are re-rendered (and the cache re-saved)


def _mouth_of(lm):
    """68-point landmarks -> mouth ellipse (cx, cy, ax, ay) in frame pixels."""
    m = lm[48:68].astype(np.float32)
    ax = (m[:, 0].max() - m[:, 0].min()) / 2 * 1.15
    ay = max((m[:, 1].max() - m[:, 1].min()) / 2 * 1.3, ax * 0.4)
    return [float(m[:, 0].mean()), float(m[:, 1].mean()), float(ax), float(ay)]


def _smooth(rows, k=5):
    """temporal mean over k frames, wrapping around (the idle loop is seamless)."""
    B = np.array(rows, np.float64)
    return np.stack([np.roll(B, s, 0) for s in range(-(k // 2), k // 2 + 1)]).mean(0)


def skin_prep(ori, mouth):
    """11-22: MuseTalk regenerates the lower face at 256x256: smooth (freckles / skin texture gone) and a bit off in colour
    (chin up to ~35% darker), so a line showed across the face where the patch ends. Away from the lips:
    1) a low-frequency colour field pulls the generated skin to the original's tone (per pixel, the mouth itself excluded
       from the measurement and filled from around it), 2) the original's fine texture replaces the generated one.
    The lips / teeth keep MuseTalk's render. This part depends on the idle frame only (cached per frame); skin_apply() does
    the per-patch half. ori: float32 HxWx3 of the patch rectangle; mouth: (cx, cy, ax, ay) in it."""
    Hh, Ww = ori.shape[:2]
    cx, cy, ax, ay = mouth
    yy, xx = np.mgrid[0:Hh, 0:Ww].astype(np.float32)
    dy = (yy - cy) / np.where(yy > cy, ay * 1.3, ay)        # a little more room below (the jaw drops when talking)
    r = np.sqrt(((xx - cx) / ax) ** 2 + dy ** 2)
    keep = 1 - np.clip((1.9 - r) / 0.8, 0, 1)[..., None]    # 0 on the lips, 1 from r 1.9 on: where the texture is swapped
    wk = 1 - np.clip((1.6 - r) / 0.4, 0, 1)                 # where the colour is measured: not the mouth
    q = 4                                                   # the colour field is smooth: 1/4 resolution
    sw, sh = max(1, Ww // q), max(1, Hh // q)
    k4 = cv2.resize(wk, (sw, sh), interpolation=cv2.INTER_AREA)[..., None]
    s1 = max(1.0, ax * 0.18 / q)
    bw, bw2 = cv2.GaussianBlur(k4, (0, 0), s1)[..., None], cv2.GaussianBlur(k4, (0, 0), s1 * 4)[..., None]
    t = np.clip(bw / 0.3, 0, 1)
    f1, f2 = t / np.maximum(bw, 1e-3), (1 - t) / np.maximum(bw2, 1e-3)

    def mean(img4):
        return f1 * cv2.GaussianBlur(img4 * k4, (0, 0), s1) + f2 * cv2.GaussianBlur(img4 * k4, (0, 0), s1 * 4)
    mo = mean(cv2.resize(ori, (sw, sh), interpolation=cv2.INTER_AREA))
    hi_o = np.clip(ori - cv2.GaussianBlur(ori, (0, 0), 1.6), -14, 14)
    return {'size': (Ww, Hh), 'sw': (sw, sh), 'k4': k4, 's1': s1, 'f1': f1, 'f2': f2, 'mo': mo + 2, 'keep': keep,
            'hik': (hi_o * keep).astype(np.float32), 'mean': None}


def skin_apply(P, gen):
    k4, s1 = P['k4'], P['s1']
    g4 = cv2.resize(gen, P['sw'], interpolation=cv2.INTER_AREA) * k4
    mg = P['f1'] * cv2.GaussianBlur(g4, (0, 0), s1) + P['f2'] * cv2.GaussianBlur(g4, (0, 0), s1 * 4)
    field = cv2.resize(np.clip(P['mo'] / (mg + 2), 0.75, 1.35), P['size'], interpolation=cv2.INTER_LINEAR)
    g = gen * field
    # g + keep*(hi_o - hi_g)  ==  g*(1-keep) + keep*blur(g) + keep*hi_o
    return g - P['keep'] * (g - cv2.GaussianBlur(g, (0, 0), 1.6)) + P['hik']


def skin_fix(ori, gen, mouth):
    return skin_apply(skin_prep(ori, mouth), gen)


def _align16(x0, y0, x1, y1, w, h):
    x0 = max(0, (x0 // 16) * 16); y0 = max(0, (y0 // 16) * 16)
    x1 = min(w, ((x1 + 15) // 16) * 16); y1 = min(h, ((y1 + 15) // 16) * 16)
    return int(x0), int(y0), int(x1), int(y1)


def read_frames(video_path):
    """11-22: decoded the way the phone's browser shows the video (BT.709, the browsers' default for untagged HD video).
    OpenCV decodes with BT.601, so every patch came out slightly redder / less green than the video around it (~2 levels
    on Holmes' face) and its edge showed as a line. Falls back to OpenCV if ffmpeg is missing or fails."""
    fps = _probe_fps(video_path)
    try:
        import subprocess
        raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', video_path, '-vsync', '0', '-vf',
                              f'scale={W}:{H}:flags=area:in_color_matrix=bt709:out_color_matrix=bt709:in_range=tv:out_range=pc',
                              '-pix_fmt', 'bgr24', '-f', 'rawvideo', '-'], capture_output=True, check=True, timeout=300).stdout
        n = len(raw) // (W * H * 3)
        if n < 2:
            raise RuntimeError('ffmpeg gave no frames')
        a = np.frombuffer(raw[:n * W * H * 3], np.uint8).reshape(n, H, W, 3)
        return [a[i].copy() for i in range(n)], fps
    except Exception as e:
        print('[engine] ffmpeg decode failed, using OpenCV:', e, flush=True)
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA))
    cap.release()
    return frames, fps


def _probe_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    cap.release()
    return float(fps)


def enc(img, q=JPEG_Q):
    return cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()


class Models:
    def __init__(self, repo, batch=8):
        os.chdir(repo)                      # MuseTalk loads ./models/... with relative paths
        sys.path.insert(0, repo)
        import torch
        from transformers import WhisperModel
        from musetalk.utils.utils import load_all_model
        from musetalk.utils.audio_processor import AudioProcessor
        from musetalk.utils.face_parsing import FaceParsing
        self.torch = torch
        self.dev = torch.device('cuda')
        self.batch = batch
        t = time.time()
        vae, unet, pe = load_all_model(unet_model_path='models/musetalkV15/unet.pth', vae_type='sd-vae',
                                       unet_config='models/musetalkV15/musetalk.json', device=self.dev)
        self.pe = pe.half().to(self.dev)
        vae.vae = vae.vae.half().to(self.dev)
        unet.model = unet.model.half().to(self.dev)
        self.vae, self.unet = vae, unet
        self.dtype = unet.model.dtype
        self.timesteps = torch.tensor([0], device=self.dev)
        self.ap = AudioProcessor(feature_extractor_path='models/whisper')
        self.whisper = WhisperModel.from_pretrained('models/whisper').to(device=self.dev, dtype=self.dtype).eval()
        self.whisper.requires_grad_(False)
        self.FaceParsing = FaceParsing
        self.fp = None
        self.fa = None
        self.gpu = torch.cuda.get_device_name(0)
        print(f'[engine] models loaded in {time.time() - t:.1f}s on {self.gpu}', flush=True)

    def chunks(self, audio_f32):
        feats = []
        for i in range(0, len(audio_f32), 30 * 16000):
            f = self.ap.feature_extractor(audio_f32[i:i + 30 * 16000], return_tensors='pt', sampling_rate=16000).input_features
            feats.append(f.to(dtype=self.dtype))
        return self.ap.get_whisper_chunk(feats, self.dev, self.dtype, self.whisper, len(audio_f32), fps=FPS,
                                         audio_padding_length_left=2, audio_padding_length_right=2)

    def faces(self, latents, chunks, idxs):
        torch = self.torch
        k = len(idxs)
        if k < self.batch:          # always run a full batch: every new batch shape costs a slow first call on the GPU
            idxs = list(idxs) + [idxs[-1]] * (self.batch - k)
            chunks = torch.cat([chunks, chunks[-1:].expand(self.batch - k, *chunks.shape[1:])], 0)
        with torch.no_grad():
            lat = latents[idxs]
            af = self.pe(chunks.to(self.dev))
            pred = self.unet.model(lat, self.timesteps, encoder_hidden_states=af).sample
            return self.vae.decode_latents(pred.to(dtype=self.vae.vae.dtype))[:k]   # uint8 BGR (k,256,256,3)


class Avatar:
    """cache: bytes from a previous to_bytes() of the same video, or None (prepare now)."""

    def __init__(self, M, video_path, cache=None):
        self.M = M
        self.frames, self.vfps = read_frames(video_path)
        self.n = len(self.frames)
        self._skin = {}                      # idle frame -> skin_prep() result (~1.6 MB each)
        if self.n < 10:
            raise RuntimeError(f'video has only {self.n} frames')
        d = None
        if cache:
            try:
                d = pickle.loads(cache)
                if d.get('n') != self.n:
                    d = None
            except Exception as e:
                print('[engine] bad cache', e, flush=True); d = None
        self.fresh = d is None
        if d is None:
            d = self._prepare()
        if 'mouth' not in d:                 # caches from before 11-22: landmarks again (seconds), latents kept
            d['mouth'] = self._mouths(); self.fresh = True
        self.coords, self.rects, self.alpha, self.mouth = d['coords'], d['rects'], d['alpha'], d['mouth']
        torch = M.torch
        self.latents = torch.cat([torch.from_numpy(l) for l in d['latents']], 0).to(M.dev, dtype=M.dtype)
        self._d = d
        if 'silent' not in d or d.get('sv') != SILENT_V:
            d['silent'] = self._silent(); d['sv'] = SILENT_V; self.fresh = True
        self.silent = d['silent']          # [(x, y, jpeg)] per frame

    def to_bytes(self):
        return pickle.dumps(self._d, protocol=4)

    def _fa(self):
        import face_alignment
        M = self.M
        if M.fa is None:
            lt = getattr(face_alignment.LandmarksType, 'TWO_D', None) or getattr(face_alignment.LandmarksType, '_2D')
            M.fa = face_alignment.FaceAlignment(lt, flip_input=False, device='cuda')
        return M.fa

    def _mouths(self):
        fa = self._fa()
        ms = []
        for fr in self.frames:
            lms = fa.get_landmarks_from_image(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            ms.append(_mouth_of(lms[0]) if lms else None)
        return self._fill_smooth(ms)

    def _fill_smooth(self, ms):
        ok = [i for i, m in enumerate(ms) if m is not None]
        if not ok:
            return [None] * self.n
        ms = [ms[i] if ms[i] is not None else ms[min(ok, key=lambda j: abs(j - i))] for i in range(self.n)]
        return [list(map(float, r)) for r in _smooth(ms)]

    def _prepare(self):
        from musetalk.utils.blending import get_image_prepare_material
        M = self.M
        torch = M.torch
        self._fa()
        if M.fp is None:
            M.fp = M.FaceParsing(left_cheek_width=90, right_cheek_width=90)
        t = time.time()
        boxes, mouths = [], []
        for fr in self.frames:
            lms = M.fa.get_landmarks_from_image(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            if not lms:
                boxes.append(None); mouths.append(None); continue
            mouths.append(_mouth_of(lms[0]))
            lm = lms[0].astype(np.int32)
            # MuseTalk get_landmark_and_bbox (bbox_shift 0): top = nose point 29 minus (chin bottom - point 29)
            half = lm[29]
            dist = np.max(lm[:, 1]) - half[1]
            top = max(0, half[1] - dist)
            boxes.append([float(np.min(lm[:, 0])), float(top), float(np.max(lm[:, 0])), float(np.max(lm[:, 1]))])
        ok = [b for b in boxes if b is not None]
        if len(ok) < self.n * 0.6:
            raise RuntimeError(f'face found in only {len(ok)}/{self.n} frames')
        # frames without a face: the nearest frame's box
        for i in range(self.n):
            if boxes[i] is None:
                j = min((j for j in range(self.n) if boxes[j] is not None), key=lambda j: abs(j - i))
                boxes[i] = boxes[j]
        B = np.array(boxes)
        k = 5      # smooth over time (wraps around: the loop is seamless) so landmark noise does not make the mouth jitter
        Bs = np.stack([np.roll(B, s, 0) for s in range(-(k // 2), k // 2 + 1)]).mean(0)
        coords, latents, rects, alpha = [], [], [], []
        for i, fr in enumerate(self.frames):
            x1, y1, x2, y2 = [int(round(v)) for v in Bs[i]]
            y2 = min(y2 + 10, H)            # v1.5 extra_margin
            x1, y1 = max(0, x1), max(0, y1)
            coords.append((x1, y1, x2, y2))
            crop = cv2.resize(fr[y1:y2, x1:x2], (256, 256), interpolation=cv2.INTER_LANCZOS4)
            with torch.no_grad():
                latents.append(M.vae.get_latents_for_unet(crop).float().cpu().numpy())
            mask, cb = get_image_prepare_material(fr, [x1, y1, x2, y2], fp=M.fp, mode='jaw')
            xs, ys = int(cb[0]), int(cb[1])
            full = np.zeros((H, W), np.uint8)       # mask placed in frame coordinates (crop box may leave the frame)
            mh, mw = mask.shape[:2]
            fx0, fy0, fx1, fy1 = max(0, xs), max(0, ys), min(W, xs + mw), min(H, ys + mh)
            full[fy0:fy1, fx0:fx1] = mask[fy0 - ys:fy1 - ys, fx0 - xs:fx1 - xs]
            ys_, xs_ = np.nonzero(full > 2)
            rx0, ry0, rx1, ry1 = _align16(xs_.min(), ys_.min(), xs_.max() + 1, ys_.max() + 1, W, H)
            a = full[ry0:ry1, rx0:rx1].astype(np.float32) / 255.0
            rects.append((rx0, ry0, rx1, ry1))
            alpha.append(a[..., None].astype(np.float16))
        print(f'[engine] avatar prepared: {self.n} frames in {time.time() - t:.1f}s, patch ~{rects[0][2]-rects[0][0]}x{rects[0][3]-rects[0][1]}', flush=True)
        return {'n': self.n, 'coords': coords, 'latents': latents, 'rects': rects, 'alpha': alpha, 'mouth': self._fill_smooth(mouths)}

    def _blend(self, idx, face):
        x1, y1, x2, y2 = self.coords[idx]
        rx0, ry0, rx1, ry1 = self.rects[idx]
        ori = self.frames[idx]
        face = cv2.resize(face, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
        region = ori[ry0:ry1, rx0:rx1]
        large = region.copy()
        fx0, fy0, fx1, fy1 = max(x1, rx0), max(y1, ry0), min(x2, rx1), min(y2, ry1)
        if fx1 > fx0 and fy1 > fy0:
            large[fy0 - ry0:fy1 - ry0, fx0 - rx0:fx1 - rx0] = face[fy0 - y1:fy1 - y1, fx0 - x1:fx1 - x1]
        a = self.alpha[idx].astype(np.float32)
        m = self.mouth[idx] if self.mouth else None
        if m is not None:
            P = self._skin.get(idx)
            if P is None:
                P = self._skin[idx] = skin_prep(region.astype(np.float32), (m[0] - rx0, m[1] - ry0, m[2], m[3]))
            large = skin_apply(P, large.astype(np.float32))
        out = region * (1 - a) + large * a
        return rx0, ry0, np.clip(out + 0.5, 0, 255).astype(np.uint8)

    def _render1(self, i, f):
        x, y, p = self._blend(i, f)
        return x, y, enc(p)

    def _render(self, idxs, faces):
        return list(POOL.map(self._render1, idxs, faces))

    def _silent(self):
        t = time.time()
        n, M = self.n, self.M
        chunks = M.chunks(np.zeros(n * 640, np.float32))
        out = []
        for b in range(0, n, M.batch):
            idxs = list(range(b, min(b + M.batch, n)))
            faces = M.faces(self.latents, chunks[b:b + len(idxs)], idxs)
            out.extend(self._render(idxs, faces))
        print(f'[engine] silent patches rendered in {time.time() - t:.1f}s', flush=True)
        return out

    def generate(self, pcm16, idxs_of, emit, cancel=None):
        """Blocking. pcm16: 16 kHz int16 (len % 640 == 0). idxs_of(k) -> idle frame index for audio frame k.
        emit(k, x, y, jpeg) for every 40 ms frame. Returns stats.
        Pipelined: the GPU works on batch b+1 while the CPU pool blends / encodes batch b; a drain thread emits in order."""
        import queue, threading
        M = self.M
        t0 = time.time()
        chunks = M.chunks(pcm16.astype(np.float32) / 32768.0)
        n = chunks.shape[0]
        st = {'first': None, 'done': 0}
        q = queue.Queue()

        def drain():
            while True:
                item = q.get()
                if item is None:
                    return
                k, fut = item
                x, y, jpg = fut.result()
                emit(k, x, y, jpg)
                st['done'] = k + 1
                if st['first'] is None:
                    st['first'] = time.time() - t0
        th = threading.Thread(target=drain, daemon=True); th.start()
        try:
            for b in range(0, n, M.batch):
                if cancel and cancel.is_set():
                    break
                ks = list(range(b, min(b + M.batch, n)))
                idxs = [idxs_of(k) % self.n for k in ks]
                faces = M.faces(self.latents, chunks[b:b + len(ks)], idxs)
                for k, i, f in zip(ks, idxs, faces):
                    q.put((k, POOL.submit(self._render1, i, f)))
        finally:
            q.put(None); th.join()
        dt = time.time() - t0
        done = st['done']
        return {'frames': done, 'sec': round(dt, 3), 'fps': round(done / dt, 1) if dt else 0, 'first_ms': int((st['first'] or 0) * 1000)}

class MockModels:
    gpu = 'mock'


class MockAvatar:
    """CPU stand-in for local testing of the page/protocol: an ellipse at a guessed mouth spot opens with the audio loudness."""
    fresh = True

    def __init__(self, M, video_path, cache=None):
        self.frames, self.vfps = read_frames(video_path)
        self.n = len(self.frames)
        self.rect = _align16(360 - 96, 700 - 64, 360 + 96, 700 + 80, W, H)
        x0, y0, x1, y1 = self.rect
        self.silent = [(x0, y0, enc(f[y0:y1, x0:x1])) for f in self.frames]

    def to_bytes(self):
        return b'mock-avatar-cache!'

    def generate(self, pcm16, idxs_of, emit, cancel=None):
        t0 = time.time()
        n = len(pcm16) // 640
        x0, y0, x1, y1 = self.rect
        for k in range(n):
            if cancel and cancel.is_set():
                break
            seg = pcm16[k * 640:(k + 1) * 640].astype(np.float32) / 32768
            o = min(1.0, float(np.sqrt((seg ** 2).mean())) * 8)
            i = idxs_of(k) % self.n
            p = self.frames[i][y0:y1, x0:x1].copy()
            cv2.ellipse(p, (96, 64 + 8), (30, max(1, int(26 * o))), 0, 0, 360, (30, 20, 140), -1)
            emit(k, x0, y0, enc(p))
            if k % 8 == 7:
                time.sleep(0.12)          # ~65 fps, like a GPU a bit faster than real time
        dt = time.time() - t0
        return {'frames': n, 'sec': round(dt, 3), 'fps': round(n / dt, 1) if dt else 0, 'first_ms': 5}
