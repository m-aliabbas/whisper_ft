from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import torch
import numpy as np
from typing import List
import pathlib

class WhisperBatchASR:
    def __init__(self, model_name="openai/whisper-medium", device="cuda"):
        self.device = device
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
        self.model.eval()

    def load_and_preprocess(self, paths: List[pathlib.Path]) -> List[np.ndarray]:
        audio_list = []
        for path in paths:
            waveform, sr = torchaudio.load(path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
            audio_list.append(waveform.squeeze(0).numpy())
        return audio_list

    def transcribe(self, paths: List[pathlib.Path], batch_size=4) -> dict:
        audio_data = self.load_and_preprocess(paths)

        results = {}
        for i in range(0, len(audio_data), batch_size):
            batch_audios = audio_data[i:i+batch_size]
            inputs = self.processor(batch_audios, sampling_rate=16000, return_tensors="pt",padding="max_length")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = self.model.generate(inputs["input_features"])

            texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

            for j, text in enumerate(texts):
                results[str(paths[i + j].name)] = text

        return results


# from pathlib import Path

# asr = WhisperBatchASR(model_name="openai/whisper-medium", device="cuda")
# audio_files = list(Path("audio_samples/").glob("*.wav"))
# import time

# t1 = time.time()
# results = asr.transcribe(audio_files, batch_size=4)
# t2 = time.time()
# print('Time',t1-t2) 
# for fname, text in results.items():
#     print(f"{fname}: {text}")
