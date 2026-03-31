import torch
print(f"Versión de Torch: {torch.__version__}")
print(f"¿CUDA disponible?: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
