# app/main.py
from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
import tempfile
import os
import soundfile
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
from beep_detection_torch import detect_beep
import traceback
import os
import socket
import time
model_name = os.getenv("MODEL_NAME", "tiny.en")
beep_check = os.getenv("beep_detection", "OFF")
app = FastAPI()

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"
print(f"Using model: {model_name} on device: {device} with compute type: {compute_type}")
model = WhisperModel(model_name, device=device, compute_type=compute_type)
host_name = socket.gethostname()
def load_audio(file_path: str):
    """Load audio file and return audio data and sample rate."""
    audio, sample_rate = soundfile.read(file_path)
    if len(audio.shape) > 1:  # Check if stereo
        audio = np.mean(audio, axis=1)  # Convert to mono
    return audio, sample_rate

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...),
    audio_tagging_time_resolution: Optional[int] = Form(4.0),
    temperature: Optional[float] = Form(0.01),
    no_speech_threshold: Optional[float] = Form(0.4)):
    # Save uploaded file temporarily

    start_time = time.time()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name 
    
    try:
        if "off" not in beep_check.lower():
            beep_result = detect_beep(tmp_path)[0]
        else:
            beep_result = False
        if beep_result:
            transcript = "DAIL TONE"
            return {
                "language": '',
                "language_probability": '',
                "text": transcript,
                "hostname":host_name,
                "execution_time":time.time() - start_time,
            }
        else:
            # Load the audio file
            wav = load_audio(tmp_path)[0]
            # print(f"Audio shape: {wav.shape}")
            if wav is None:
                return {"text": "Error loading audio file.", "hostname":host_name,
                "execution_time":time.time() - start_time,}
            if np.sqrt(np.mean(wav ** 2)) < 0.0017:
                return {"text": "", "hostname":host_name,
                "execution_time":time.time() - start_time,}

            segments, info = model.transcribe(tmp_path, beam_size=5,
            best_of=1,
            vad_filter=True,         # <-- Whisper strips silence itself)
            language="en",          # <-- Specify the language
            )
            transcript = "".join(f"{seg.text}" for seg in segments)
        
            return {
                "language": info.language,
                "language_probability": info.language_probability,
                "text": transcript,
                 "hostname":host_name,
                "execution_time":time.time() - start_time,
            }
    
    except Exception:
        traceback.print_exc()
        return {
            "language": "",
            "language_probability": "",
            "text": "",
             "hostname":host_name,
                "execution_time":time.time() - start_time
        }

    finally:
        os.remove(tmp_path)
        

@app.get("/health")
def health_check():
    return {"status": "ok"}