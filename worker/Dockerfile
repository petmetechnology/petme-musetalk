# petmeai.com 精确口型 worker for RunPod Serverless (MuseTalk 1.5). Built by .github/workflows/image.yml and pushed to GHCR (public).
# Everything (code + weights) is baked in, so a cold worker only loads the models (~15 s) and the character's avatar.
# No secrets in here: each job brings its own one-time session secret (functions/petmt.js).
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 MUSETALK_DIR=/opt/MuseTalk HF_HUB_DISABLE_TELEMETRY=1
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends ffmpeg git wget ca-certificates libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/* \
 && wget -q -O /usr/local/bin/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
 && chmod +x /usr/local/bin/cloudflared
RUN python -c "import torch,torchvision;print(f'torch=={torch.__version__}\ntorchvision=={torchvision.__version__}')" > /tmp/constraints.txt \
 && pip install --no-cache-dir -q -c /tmp/constraints.txt "numpy<2" diffusers==0.30.2 transformers==4.39.2 accelerate==0.28.0 \
    "huggingface_hub>=0.23,<0.31" librosa==0.11.0 soundfile einops omegaconf opencv-python-headless==4.9.0.80 \
    face_alignment==1.4.1 fastapi==0.115.6 "uvicorn[standard]==0.32.1" httpx==0.27.2 "runpod>=1.7,<2"
RUN git clone --depth 1 https://github.com/TMElyralab/MuseTalk.git /opt/MuseTalk && rm -rf /opt/MuseTalk/.git
WORKDIR /opt/MuseTalk
RUN mkdir -p models/musetalkV15 models/sd-vae models/whisper models/face-parse-bisent && python - <<'PY'
from huggingface_hub import hf_hub_download as d
d('TMElyralab/MuseTalk', 'musetalkV15/musetalk.json', local_dir='models')
d('TMElyralab/MuseTalk', 'musetalkV15/unet.pth', local_dir='models')
for f in ['config.json', 'diffusion_pytorch_model.bin']: d('stabilityai/sd-vae-ft-mse', f, local_dir='models/sd-vae')
for f in ['config.json', 'pytorch_model.bin', 'preprocessor_config.json']: d('openai/whisper-tiny', f, local_dir='models/whisper')
PY
RUN wget -q -O models/face-parse-bisent/resnet18-5c106cde.pth https://download.pytorch.org/models/resnet18-5c106cde.pth
COPY weights/79999_iter.pth models/face-parse-bisent/79999_iter.pth
# face_alignment fetches its detector + FAN weights on first use: do it now (CPU is fine for the download)
RUN python -c "import face_alignment as f;t=getattr(f.LandmarksType,'TWO_D',None) or f.LandmarksType._2D;f.FaceAlignment(t,device='cpu')"
COPY worker/engine.py worker/handler.py /opt/worker/
WORKDIR /opt/worker
CMD ["python", "-u", "handler.py"]
