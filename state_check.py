import torch

print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    idx = torch.cuda.current_device()
    print("Current device:", idx, torch.cuda.get_device_name(idx))
    print("Allocated (MB):", torch.cuda.memory_allocated(idx) / 1024**2)
    
