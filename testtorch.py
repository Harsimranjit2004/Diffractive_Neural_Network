import torch

# Check if CUDA (GPU) is available
if torch.cuda.is_available():
    print("✅ GPU is available!")
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("CUDA Version:", torch.version.cuda)
else:
    print("❌ GPU is NOT available. Using CPU.")
