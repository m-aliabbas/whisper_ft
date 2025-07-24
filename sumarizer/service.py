import bentoml
from bentoml.io import File, JSON
import tempfile
from typing import List
import pathlib
@bentoml.service(
    image=bentoml.images.Image(python_version="3.10").python_packages(
        "torch", 
        "transformers", 
        "torchaudio", 
        "ffmpeg-python"
    ),
    resources={"gpu": 1},
    workers=2
)
class Transcription:
    def __init__(self) -> None:
        import torch
        from transformers import pipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3",
            device=0 if device == "cuda" else -1,
        )

    @bentoml.api(batchable=True)
    def transcribe(self, audio_files: List[pathlib.Path]) -> dict:
        import torchaudio

        waveform, sr = torchaudio.load(audio_file)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        result = self.pipeline(waveform.squeeze(0).numpy())
        
        return {"text": result}
    
    
        