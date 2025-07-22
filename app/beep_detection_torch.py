import torch
import numpy as np
import scipy.io.wavfile as wav
import traceback

def detect_beep(audio_file, threshold=3900, min_beep_length=1.0, device='cpu'):
    try:
        # Read audio file on CPU
        
        sample_rate, audio_data = wav.read(audio_file)
        
        # Convert to float32 and handle stereo by averaging channels
        audio_data = audio_data.astype(np.float32)
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Move data to PyTorch tensor and GPU
        audio_tensor = torch.from_numpy(audio_data).float().to(device)
        
        # Compute envelope (absolute value)
        envelope = torch.abs(audio_tensor)
        
        # Convolution for smoothing
        window_size = max(int(0.05 * sample_rate), 1)
        window = torch.ones(1, 1, window_size, device=device) / window_size
        
        # Reshape envelope for conv1d: [length] -> [1, 1, length]
        envelope = envelope.unsqueeze(0).unsqueeze(0)
        smoothed_envelope = torch.conv1d(envelope, window, padding=window_size//2).squeeze()
        
        # Identify regions above threshold
        above_thresh = smoothed_envelope > threshold
        
        # Find transitions using diff
        transitions = torch.diff(above_thresh.to(torch.int8))
        start_indices = (transitions == 1).nonzero(as_tuple=False).squeeze()
        end_indices = (transitions == -1).nonzero(as_tuple=False).squeeze()
        
        # Ensure start_indices and end_indices are 1D tensors
        if start_indices.dim() == 0:
            start_indices = start_indices.unsqueeze(0)
        if end_indices.dim() == 0:
            end_indices = end_indices.unsqueeze(0)
        
        # Handle edge cases
        if above_thresh[0]:
            start_indices = torch.cat((torch.tensor([0], device=device), start_indices))
        
        if above_thresh[-1]:
            end_indices = torch.cat((end_indices, torch.tensor([len(above_thresh)], device=device)))
        
        # Ensure starts and ends are paired
        if start_indices.numel() > end_indices.numel():
            end_indices = torch.cat((end_indices, torch.tensor([len(above_thresh)], device=device)))
        
        
        # Filter beeps by minimum length
        durations = (end_indices - start_indices).float() / sample_rate
        valid_beeps = durations > min_beep_length
        
        # Convert to CPU for final output
        beep_events = [(start.item() / sample_rate, end.item() / sample_rate)
                       for start, end in zip(start_indices[valid_beeps].cpu(), end_indices[valid_beeps].cpu())]
        
        return (True, beep_events) if beep_events else (False, [])
    
    except Exception as e:
        # Print full stack trace
        traceback.print_exc()
        return (False, [])