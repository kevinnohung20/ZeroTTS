import io
import os
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from huggingface_hub import snapshot_download
from zerotts import ZeroTTS

app = FastAPI(title="ZeroTTS API")

print("Downloading/Checking ZeroTTS model files...")
# Tải trực tiếp file thật vào thư mục /app/model_cache (không dùng symlink để tránh lỗi ONNX)
model_path = snapshot_download(
    repo_id="zeroweight-ai/ZeroTTS",
    local_dir="/app/model_cache",
    local_dir_use_symlinks=False
)

print(f"Loading ZeroTTS model from {model_path}...")
tts = ZeroTTS.from_pretrained(model_path)
print("Model loaded successfully!")

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

@app.get("/voices")
def get_voices():
    return {"voices": tts.list_voices()}

@app.get("/health")
def health():
    return {"status": "ok"}
