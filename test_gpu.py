#!/usr/bin/env python3
"""
test_gpu.py - Diagnóstico de GPU para Eagle2.5-8B en RTX 5070 Ti (sm_120)
Ejecutar: CUDA_LAUNCH_BLOCKING=1 python test_gpu.py
"""
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
print(f"\n=== Entorno ===")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Capability: {torch.cuda.get_device_capability(0)}")
print(f"Arch list: {torch.cuda.get_arch_list()}")

print("\n=== Test 1: tensor básico float32 en GPU ===")
x = torch.randn(4, 4).cuda()
y = x @ x
print(f"OK: {y.shape}")

print("\n=== Test 2: tensor bfloat16 en GPU ===")
x = torch.randn(4, 4, dtype=torch.bfloat16).cuda()
y = x @ x
print(f"OK bfloat16: {y.shape}")

print("\n=== Test 3: tensor float16 en GPU ===")
x = torch.randn(4, 4, dtype=torch.float16).cuda()
y = x @ x
print(f"OK float16: {y.shape}")

print("\n=== Test 4: embedding en GPU con bfloat16 ===")
emb = torch.nn.Embedding(1000, 128).cuda().to(torch.bfloat16)
idx = torch.randint(0, 1000, (4, 16)).cuda()
out = emb(idx)
print(f"OK embedding bfloat16: {out.shape}")

print("\n=== Test 5: LayerNorm bfloat16 en GPU ===")
ln = torch.nn.LayerNorm(128).cuda().to(torch.bfloat16)
x = torch.randn(4, 16, 128, dtype=torch.bfloat16).cuda()
out = ln(x)
print(f"OK LayerNorm bfloat16: {out.shape}")

print("\n=== Test 6: SDPA bfloat16 en GPU ===")
q = torch.randn(1, 8, 16, 64, dtype=torch.bfloat16).cuda()
k = torch.randn(1, 8, 16, 64, dtype=torch.bfloat16).cuda()
v = torch.randn(1, 8, 16, 64, dtype=torch.bfloat16).cuda()
out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
print(f"OK SDPA bfloat16: {out.shape}")

print("\n=== Test 7: carga del modelo Eagle2.5-8B con eager ===")
from transformers import AutoModel, AutoProcessor
model = AutoModel.from_pretrained(
    "nvidia/Eagle2.5-8B",
    trust_remote_code=True,
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
)
model = model.to("cuda")
print("OK: modelo cargado en GPU con eager+bfloat16")

print("\n=== Test 8: inferencia simple con imagen en blanco ===")
from PIL import Image
import numpy as np
processor = AutoProcessor.from_pretrained("nvidia/Eagle2.5-8B", trust_remote_code=True, use_fast=True)
processor.tokenizer.padding_side = "left"

img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": "Describe esta imagen."},
    ],
}]
text_list = [processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)]
image_inputs, video_inputs = processor.process_vision_info(messages)
inputs = processor(text=text_list, images=image_inputs, videos=video_inputs, return_tensors="pt", padding=True)
inputs = inputs.to("cuda")

with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
result = processor.batch_decode(out, skip_special_tokens=True)
print(f"OK inferencia: {result}")

print("\n=== Todos los tests pasaron. GPU funciona con Eagle2.5-8B ===")
