"""Zomi (Tedim) TTS website backend — Meta MMS facebook/mms-tts-ctd."""
import os
import re
import shutil
import subprocess
import tempfile

# Sanitize proxy env vars (httpx crashes on bracketed IPv6 entries)
for var in ("no_proxy", "NO_PROXY"):
    if os.environ.get(var):
        cleaned = [e.strip() for e in os.environ[var].split(",") if "[" not in e and "]" not in e]
        os.environ[var] = ",".join(cleaned)

import numpy as np
import scipy.io.wavfile as wavfile
import torch
from transformers import VitsModel, AutoTokenizer
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

try:
    import parselmouth
    from parselmouth.praat import call
    HAS_PARSELMOUTH = True
except ImportError:
    HAS_PARSELMOUTH = False

MODEL_ID = "facebook/mms-tts-ctd"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_CHARS = 2000

print(f"Loading {MODEL_ID} on {DEVICE}...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = VitsModel.from_pretrained(MODEL_ID).to(DEVICE)
model.eval()
SR = model.config.sampling_rate
print(f"Model ready. Sample rate: {SR} Hz", flush=True)


def split_sentences(text: str):
    parts = re.split(r"(?<=[.!?\n])\s+", text)
    return [p.strip() for p in parts if p.strip()] or [text.strip()]


def female_shift(audio: np.ndarray) -> np.ndarray:
    if not HAS_PARSELMOUTH:
        return audio
    try:
        sound = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=SR)
        shifted = call(sound, "Change gender", 75.0, 600.0, 1.15, 215.0, 1.0, 1.0)
        return np.asarray(shifted.as_array()).squeeze().astype(np.float32)
    except Exception as e:
        print(f"Voice shift failed, using normal: {e}", flush=True)
        return audio


def synthesize(text: str, female: bool = False) -> np.ndarray:
    chunks, gap = [], np.zeros(int(SR * 0.08), dtype=np.float32)
    for sent in split_sentences(text):
        inputs = tokenizer(sent, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            seg = model(**inputs).waveform.squeeze().cpu().numpy().astype(np.float32)
        peak = np.abs(seg).max()
        if peak > 1e-6:
            seg = seg / peak * 0.95
        chunks += [seg, gap]
    audio = np.concatenate(chunks).astype(np.float32)
    if female:
        audio = female_shift(audio).astype(np.float32)
    return np.clip(audio, -1.0, 1.0)


def to_download_file(audio: np.ndarray):
    """Write WAV, convert to MP3 with ffmpeg when available. Returns (path, media_type, ext)."""
    wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(wav_fd)
    wavfile.write(wav_path, SR, (audio * 32767).astype(np.int16))
    if shutil.which("ffmpeg"):
        mp3_fd, mp3_path = tempfile.mkstemp(suffix=".mp3")
        os.close(mp3_fd)
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-codec:a", "libmp3lame", "-b:a", "192k", mp3_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if r.returncode == 0:
            os.remove(wav_path)
            return mp3_path, "audio/mpeg", "mp3"
        os.remove(mp3_path)
    return wav_path, "audio/wav", "wav"


class SpeakRequest(BaseModel):
    text: str
    voice: str = "normal"  # "normal" | "female"


app = FastAPI(title="Zomi (Tedim) TTS")


@app.get("/", response_class=HTMLResponse)
def index():
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "static", "index.html"), encoding="utf-8") as f:
        return f.read()


@app.get("/api/health")
def health():
    return {"ok": True, "model": MODEL_ID, "device": DEVICE,
            "female_shift": HAS_PARSELMOUTH}


@app.post("/api/speak")
def speak(req: SpeakRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Type some Tedim text first. / Tedim lai gelh in.")
    if len(text) > MAX_CHARS:
        raise HTTPException(status_code=400, detail=f"Text too long (max {MAX_CHARS} characters).")
    try:
        audio = synthesize(text, female=(req.voice == "female"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Speech generation failed: {e}")
    path, media_type, ext = to_download_file(audio)
    return FileResponse(path, media_type=media_type, filename=f"zomi-tts.{ext}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
