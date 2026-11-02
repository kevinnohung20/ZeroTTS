import os
# Khóa 4 luồng ở mức hệ điều hành (Đã đủ để giới hạn CPU, không cần dùng thư viện torch)
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["ONNXRUNTIME_NUM_THREADS"] = "4"

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

# Mở CORS tường minh
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://nav.hoangphucshop.com",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5500"
    ],
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
        chunk_buffer = b""
        # Tăng CHUNK_SIZE lên 24576 bytes (~250ms âm thanh). Giảm chi phí mạng, chống vấp.
        CHUNK_SIZE = 24576 

        for chunk in tts.synthesize_stream(req.text, voice=req.voice):
            c = np.asarray(chunk).squeeze()
            
            if c.dtype == np.int16:
                c = c.astype(np.float32) / 32768.0
            else:
                c = c.astype(np.float32)
                c = np.clip(c, -1.0, 1.0)
            
            chunk_buffer += c.tobytes()

            while len(chunk_buffer) >= CHUNK_SIZE:
                yield chunk_buffer[:CHUNK_SIZE]
                chunk_buffer = chunk_buffer[CHUNK_SIZE:]
        
        if chunk_buffer:
            yield chunk_buffer

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
