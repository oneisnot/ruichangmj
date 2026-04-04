import torch
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    # 进行一个小张量运算验证内核
    x = torch.randn(2, 2).cuda()
    y = x @ x
    print("✅ Tensor op successful!")
