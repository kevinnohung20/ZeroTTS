import os
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["ONNXRUNTIME_NUM_THREADS"] = "8"

import io
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from huggingface_hub import snapshot_download
from zerotts import ZeroTTS

app = FastAPI(title="ZeroTTS API")

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
        # Trả về các chunk âm thanh PCM raw hoặc wav bytes liên tục
        for chunk in tts.synthesize_stream(req.text, voice=req.voice):
            yield chunk.tobytes()

    return StreamingResponse(audio_generator(), media_type="application/octet-stream")

@app.get("/health")
def health():
    return {"status": "ok"}
