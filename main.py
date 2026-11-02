import io
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from zerotts import ZeroTTS

app = FastAPI(title="ZeroTTS API")

print("Loading ZeroTTS model...")
tts = ZeroTTS.from_pretrained("zeroweight-ai/ZeroTTS")
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
