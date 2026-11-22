# 精确口型 worker（RunPod Serverless，MuseTalk 1.5）

petmeai.com 的“精确口型”。一次通话 = 一个 RunPod 任务，由 Firebase 函数 `petMt`（`functions/petmt.js`）提交。用的是按需起停的 flex worker，按秒计费，没人用时缩到 0 台。

## 链路
页面照旧自己做语音识别、大模型和 TTS，只有嘴换成 GPU 生成的：
1. 页面 `petMt start` → RunPod `/run`，任务输入是 `{sid, secret, key, video, fn}`。
2. worker 下载这个角色的待机视频，加载 avatar：先从 `fn?op=cache_get` 取缓存，没有就现做（约 1 分钟），做完 `cache_put` 存到 bucket 的 `mt/<akey>/avatar.bin`（私有）。
3. 起 FastAPI（每个 worker 起一次）和 Cloudflare 临时隧道，然后 `fn?op=ready {url}`，这一步扣第 1 分钟。之后每分钟 `lease` 扣 1 分钟，额度用完或用户关掉就返回 `stop`。
4. 页面 `petMt poll` 拿到 `wss://…trycloudflare.com/ws?k=<key>`，直接连到 worker。worker 先发 hello 和每个待机帧的静音嘴部补丁；之后页面每说一句，就把 16k PCM 连同“这句开口时屏幕上的待机帧号 i0”发过去，worker 按第 k 个 40ms 对应待机帧 `round(i0 + k·fps/25)` 渲染嘴部补丁（720×1280 坐标的 JPEG）发回。页面让音频晚 `MT_LEAD`=0.45 秒开始，用 WebGL 把补丁贴在待机层上（动作片段不贴，动作都在说完话后播）。
5. 页面挂断、切到后台：`petMt stop` → RunPod `/cancel`。页面直接消失时，worker 在 25 秒没有客户端后自己结束。

协议细节写在 `handler.py` 顶部的注释里。

## 文件
- `Dockerfile`：pytorch 2.1.2 / CUDA 11.8 运行时，MuseTalk 代码、全部权重、face_alignment 权重、cloudflared 都打进镜像，所以冷启动只需加载模型。**镜像里没有任何密钥**。
- `engine.py`：`Models`（每个 worker 加载一次）、`Avatar`（一个角色：准备、静音补丁、`generate`），以及 `MockModels`/`MockAvatar`（本机测试用）。
- `handler.py`：RunPod handler、WebSocket 服务、隧道、计费心跳。
- `github/image.yml`：GitHub Actions 构建并推送 `ghcr.io/<user>/petme-musetalk:latest`。
- `push_image_repo.sh`：建立或更新公开仓库 `<user>/petme-musetalk`，里面只放镜像需要的文件。

## 本机测试（没有 GPU）
```
cd worker && MOCK=1 LOCAL=1 PORT=8765 PYTHONIOENCODING=utf-8 python handler.py ../../testcat/girl/idle.mp4
```
页面在 localhost 打开（`python -m http.server`），设 `window.__mtLocal = 'ws://127.0.0.1:8765'`，再调用 `__tc.mt.start()`，就会跳过 RunPod 直接连本机。模拟引擎只画一个随音量张合的椭圆。测试脚本是会话临时目录的 `cdp_mt.mjs`。

## 上线步骤
1. `GH_TOKEN=… bash worker/push_image_repo.sh <github 用户名>`，等 Actions 构建完（约 20–30 分钟）。第一次推送后，在 GitHub 的 Packages 设置里把 `petme-musetalk` 改成 Public，RunPod 才能匿名拉取。
2. RunPod：新建 Serverless endpoint，镜像 `ghcr.io/<user>/petme-musetalk:latest`；GPU 选 24 GB（3090 / A5000 / 4090）；Active workers 0，Max workers 3；Idle timeout 5 秒；开启 FlashBoot；容器硬盘 30 GB。
3. `functions/.env` 加 `RUNPOD_API_KEY`、`RUNPOD_MT_ENDPOINT`，然后部署 `petMt`。

## 费用（11-21 估算）
- RunPod 24 GB flex 约 $0.0002/秒，也就是约 $0.7–0.9/小时，只在通话时计费。另外每次启动有约 1 分钟准备时间，也要算钱。
- 向用户收每分钟 $0.02（约每小时 $1.2），从 VIP 额度扣（`functions/petvip.js` 的 `RATES.mtMin`）；每轮对话另外照常收费。
