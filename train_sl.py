"""
Supervised Learning Training Script for Ruichang Mahjong AI
============================================================
从 .npz 专家数据集中读取 (State, Action) 对，
使用交叉熵损失监督训练 ResNet 策略网络。

支持: DirectML (AMD GPU) -> CPU 自动降级
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from resnet_model import MahjongPolicyResNet, count_parameters


# ============================================================
# 设备自动检测 (DirectML -> CPU 优雅降级)
# ============================================================
def get_device():
    """优先尝试 DirectML (AMD GPU), 失败则降级 CPU"""
    try:
        import torch_directml
        device = torch_directml.device()
        # 快速验证: 在设备上做一次小运算
        test_tensor = torch.randn(2, 2).to(device)
        _ = test_tensor @ test_tensor
        print(f"🚀 已激活 AMD GPU 加速 (DirectML)")
        print(f"   设备: {device}")
        return device
    except Exception as e:
        print(f"⚠️  DirectML 不可用 ({e})")
        print(f"   降级使用 CPU 训练 (5800X 全核)")
        return torch.device("cpu")


# ============================================================
# 数据集定义
# ============================================================
class MahjongExpertDataset(Dataset):
    """从 .npz 文件加载专家轨迹数据"""
    
    def __init__(self, npz_path):
        print(f"📂 加载数据集: {npz_path}")
        data = np.load(npz_path)
        self.states = torch.from_numpy(data["S"]).float()   # [N, 14, 30]
        self.actions = torch.from_numpy(data["A"]).long()    # [N]
        print(f"   样本总量: {len(self.states):,}")
        print(f"   特征维度: {self.states.shape}")
        
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        state = self.states[idx]            # [14, 30]
        action = self.actions[idx]          # scalar
        legal_mask = (state[13] > 0).float()  # Channel 13 = Legal Actions Mask
        return state, action, legal_mask


# ============================================================
# 训练主循环
# ============================================================
def train(
    dataset_path="ruichang_expert_v1.npz",
    epochs=50,
    batch_size=512,
    learning_rate=1e-3,
    val_split=0.1,
    save_dir="checkpoints"
):
    device = get_device()
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载数据
    dataset = MahjongExpertDataset(dataset_path)
    
    # 划分训练集/验证集
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False
    )
    
    print(f"\n📊 数据划分:")
    print(f"   训练集: {train_size:,} 条")
    print(f"   验证集: {val_size:,} 条")
    print(f"   Batch Size: {batch_size}")
    print(f"   训练 Batches/Epoch: {len(train_loader)}")
    
    # 构建模型
    model = MahjongPolicyResNet(
        in_channels=14, num_actions=30,
        hidden_channels=256, num_res_blocks=5
    ).to(device)
    
    total_params = count_parameters(model)
    print(f"\n🧠 模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # 损失函数 & 优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # 训练循环
    best_val_acc = 0.0
    print(f"\n{'='*60}")
    print(f"🔥 开始训练! (Epochs: {epochs}, LR: {learning_rate})")
    print(f"{'='*60}")
    
    for epoch in range(1, epochs + 1):
        # --- 训练阶段 ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        epoch_start = time.time()
        
        for batch_idx, (states, actions, masks) in enumerate(train_loader):
            states = states.to(device)
            actions = actions.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            logits = model(states, legal_mask=masks)
            loss = criterion(logits, actions)
            loss.backward()
            
            # 梯度裁剪 (防止梯度爆炸)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item() * states.size(0)
            preds = logits.argmax(dim=-1)
            train_correct += (preds == actions).sum().item()
            train_total += states.size(0)
        
        scheduler.step()
        
        avg_train_loss = train_loss / train_total
        train_acc = train_correct / train_total * 100
        
        # --- 验证阶段 ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for states, actions, masks in val_loader:
                states = states.to(device)
                actions = actions.to(device)
                masks = masks.to(device)
                
                logits = model(states, legal_mask=masks)
                loss = criterion(logits, actions)
                
                val_loss += loss.item() * states.size(0)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == actions).sum().item()
                val_total += states.size(0)
        
        avg_val_loss = val_loss / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1) * 100
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        # 打印进度
        print(f"Epoch [{epoch:3d}/{epochs}] "
              f"| Train Loss: {avg_train_loss:.4f} Acc: {train_acc:5.1f}% "
              f"| Val Loss: {avg_val_loss:.4f} Acc: {val_acc:5.1f}% "
              f"| LR: {current_lr:.2e} "
              f"| {epoch_time:.1f}s")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(save_dir, "best_policy.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': avg_val_loss,
            }, best_path)
            print(f"   💾 新最佳! 验证准确率 {val_acc:.1f}% -> 已保存 {best_path}")
    
    # 保存最终模型
    final_path = os.path.join(save_dir, "final_policy.pth")
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'val_acc': val_acc,
    }, final_path)
    
    print(f"\n{'='*60}")
    print(f"🏆 训练完成!")
    print(f"   最佳验证准确率: {best_val_acc:.1f}%")
    print(f"   最佳模型路径: {os.path.join(save_dir, 'best_policy.pth')}")
    print(f"   最终模型路径: {final_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    # 默认用现有数据集做第一轮快速验证
    dataset_file = "ruichang_expert_v1.npz"
    
    if not os.path.exists(dataset_file):
        print(f"❌ 找不到数据集 {dataset_file}！")
        print(f"   请先运行 `python ruichang_mj_sim.py` 生成数据集。")
        sys.exit(1)
    
    train(
        dataset_path=dataset_file,
        epochs=50,
        batch_size=512,
        learning_rate=1e-3,
    )
