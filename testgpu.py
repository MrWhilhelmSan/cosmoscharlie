import torch
import time
from flash_attn import flash_attn_func

# 1. Verificar hardware
print(f"--- Reporte para {torch.cuda.get_device_name(0)} ---")
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"Memoria total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# 2. Test de Flash Attention (Simulación de carga de visión)
q = torch.randn((2, 128, 8, 64), dtype=torch.float16, device="cuda")
k = torch.randn((2, 128, 8, 64), dtype=torch.float16, device="cuda")
v = torch.randn((2, 128, 8, 64), dtype=torch.float16, device="cuda")

start = time.time()
out = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=True)
end = time.time()

print(f"Flash Attention ejecutado en: {(end - start) * 1000:.4f} ms")
print("---------------------------------------")
