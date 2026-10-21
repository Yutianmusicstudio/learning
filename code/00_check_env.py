"""检查本地环境：Python、PyTorch、以及 Mac GPU（MPS）是否可用。"""

import sys

print(f"Python: {sys.version.split()[0]}")

try:
    import torch
except ImportError:
    print("❌ 未安装 PyTorch，运行: uv pip install torch")
    sys.exit(1)

print(f"PyTorch: {torch.__version__}")

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Mac GPU (MPS) 可用")
else:
    device = torch.device("cpu")
    print("⚠️ MPS 不可用，将使用 CPU（阶段 0–3 也够用）")

# 在 GPU 上做一次矩阵乘法，确认真的能跑
a = torch.randn(1000, 1000, device=device)
b = torch.randn(1000, 1000, device=device)
c = a @ b
print(f"✅ 在 {device} 上完成 1000×1000 矩阵乘法，结果形状 {tuple(c.shape)}")
