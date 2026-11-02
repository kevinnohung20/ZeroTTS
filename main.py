import os
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["ONNXRUNTIME_NUM_THREADS"] = "8"

import io
import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import snapshot_download
from zerotts import ZeroTTS

app = FastAPI(title="ZeroTTS API")

# Mở CORS toàn diện và expose header X-Sample-Rate cho Web JS đọc
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Sample-Rate", "Content-Type"],
)

model_path = snapshot_download(
    repo_id="zeroweight-ai/ZeroTTS",
    local_dir="/app/model_cache",
)

tts = ZeroTTS.from_pretrained(model_path)

class TTSRequest(BaseModel):
    text: str
    voice: str = "maichi"

@app.post("/synthesize")
def synthesize_speech(req: TTSRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    try:
        audio = tts.synthesize(req.text, voice=req.voice)
        buffer = io.BytesIO()
        sf.write(buffer, audio.squeeze(), tts.sample_rate, format="WAV")
        buffer.seek(0)
        return Response(content=buffer.read(), media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/synthesize_stream")
def synthesize_stream(req: TTSRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    def audio_generator():
        for chunk in tts.synthesize_stream(req.text, voice=req.voice):
            # 1. Ép phẳng về mảng 1 chiều
            c = np.asarray(chunk).squeeze()
            
            # 2. Xử lý chuẩn hóa nếu chunk ở dạng int16 hoặc float
            if c.dtype == np.int16:
                c = c.astype(np.float32) / 32768.0
            else:
                c = c.astype(np.float32)
                # Giới hạn biên độ [-1.0, 1.0] triệt tiêu hoàn toàn rè/bể tiếng (clipping)
                c = np.clip(c, -1.0, 1.0)
            
            # 3. Xuất ra byte float32 chuẩn C-contiguous
            yield c.tobytes()

    return StreamingResponse(
        audio_generator(),
        media_type="application/octet-stream",
        headers={"X-Sample-Rate": str(tts.sample_rate)}
    )

@app.get("/voices")
def get_voices():
    return {"voices": tts.list_voices()}

@app.get("/health")
def health():
    return {"status": "ok", "sample_rate": getattr(tts, "sample_rate", 24000)}
