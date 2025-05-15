import torch
from flash_attn.flash_attn_interface import flash_attn_func

q = torch.randn(2, 128, 8, 64, device='cuda', dtype=torch.float16, requires_grad=True)
k = q.clone()
v = q.clone()
cu_seqlens = torch.arange(0, 256 + 1, step=128, dtype=torch.int32, device='cuda')
output = flash_attn_func(q, k, v, cu_seqlens, cu_seqlens, 128, 128, 0.0, False)
print("✅ FlashAttention kernel executed successfully.")
