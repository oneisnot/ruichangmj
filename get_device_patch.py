# 临时 monkey-patch: 覆盖 train_sl.get_device 使其优先使用 CUDA
import train_sl
import torch

def get_device_cuda():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"🚀 已激活 CUDA GPU: {torch.cuda.get_device_name(0)}")
        return device
    print("⚠️  CUDA 不可用，降级 CPU")
    return torch.device("cpu")

train_sl.get_device = get_device_cuda
print("✅ get_device patch applied")
