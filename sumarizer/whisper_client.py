import torch
from transformers import pipeline
import time
import tempfile
import shutil
from pathlib import Path
pipeline = pipeline(
    task="automatic-speech-recognition",
    model="openai/whisper-medium",
    torch_dtype=torch.float16,
    device=0
)

audio_path = Path("./audio_samples/audio.wav")
input_paths = [audio_path] * 8  # repeat same audio for test
temp_paths = []

for audio_path in input_paths:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copy(audio_path, tmp.name)
        temp_paths.append(tmp.name)  # Only storing the name

t1 = time.time()
res = pipeline(temp_paths,batch_size=16)
zipped = list(zip(temp_paths, res))
t2 = time.time()  
print(t2-t1,zipped)