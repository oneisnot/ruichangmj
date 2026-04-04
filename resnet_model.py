"""
ResNet Policy Network for Ruichang Mahjong AI
==============================================
轻量级残差卷积策略网络，用于从 14x30 的棋盘特征张量中
推断出最优弃牌动作（30 维 Softmax 分类）。

架构: 5 层 ResBlock, 256 通道宽度
设计目标: 推理延迟 < 1ms, 参数量 < 5M
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """标准残差块: Conv1d -> BN -> ReLU -> Conv1d -> BN -> Skip Add -> ReLU"""
    
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip Connection (残差跳跃)
        return F.relu(out)


class MahjongPolicyResNet(nn.Module):
    """
    麻将策略残差网络
    
    Input:  [Batch, 14, 30]  (14通道 x 30牌种)
    Output: [Batch, 30]      (30种牌的打出概率 logits)
    
    Architecture:
        Input Conv (14 -> 256) -> 5x ResBlock(256) -> Global Pool -> FC(256->256) -> FC(256->30)
    """
    
    def __init__(self, in_channels=16, num_actions=30, hidden_channels=256, num_res_blocks=5):
        super().__init__()
        
        # 输入卷积层: 将 14 通道特征提升到 256 维隐藏空间
        self.input_conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # 残差骨干网络: 5 层堆叠
        self.res_blocks = nn.Sequential(
            *[ResBlock(hidden_channels) for _ in range(num_res_blocks)]
        )
        
        # 策略头 (Policy Head)
        self.policy_head = nn.Sequential(
            nn.Conv1d(hidden_channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),            # [B, 32, 30] -> [B, 960]
            nn.Linear(32 * 30, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_actions)  # -> [B, 30] raw logits
        )
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, legal_mask=None):
        """
        前向传播
        
        Args:
            x: [B, 14, 30] 状态张量
            legal_mask: [B, 30] 合法动作掩码 (可选, 1=合法, 0=非法)
            
        Returns:
            logits: [B, 30] 动作 logits (未经 softmax)
        """
        # 特征提取
        out = self.input_conv(x)
        out = self.res_blocks(out)
        
        # 策略输出
        logits = self.policy_head(out)
        
        # 如果提供了合法动作掩码，将非法动作的 logit 设为极小值
        if legal_mask is not None:
            logits = logits.masked_fill(legal_mask == 0, float('-inf'))
        
        return logits
    
    def predict_action(self, x, legal_mask=None, temperature=1.0):
        """
        推理模式: 返回最优动作 (greedy) 或按温度采样
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, legal_mask)
            if temperature <= 0:
                # Greedy: 直接取 argmax
                return torch.argmax(logits, dim=-1)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                return torch.multinomial(probs, 1).squeeze(-1)


def count_parameters(model):
    """统计模型可训练参数总量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 模型自检
    model = MahjongPolicyResNet()
    total_params = count_parameters(model)
    
    print("=" * 50)
    print("🧠 Mahjong Policy ResNet 架构摘要")
    print("=" * 50)
    print(f"输入维度: [Batch, 14, 30]")
    print(f"输出维度: [Batch, 30]")
    print(f"ResBlock 层数: 5")
    print(f"隐藏通道宽度: 256")
    print(f"可训练参数总量: {total_params:,} ({total_params / 1e6:.2f}M)")
    print("=" * 50)
    
    # 模拟一个 batch 的前向传播
    dummy_input = torch.randn(8, 14, 30)  # batch=8
    dummy_mask = torch.ones(8, 30)
    dummy_mask[:, 29] = 0  # 假设发财不能打
    
    logits = model(dummy_input, dummy_mask)
    print(f"\n前向传播测试:")
    print(f"  输入 shape: {dummy_input.shape}")
    print(f"  输出 shape: {logits.shape}")
    print(f"  输出 logits 样例 (第1条): {logits[0][:5].detach().numpy()}...")
    
    # 推理测试
    action = model.predict_action(dummy_input, dummy_mask, temperature=0)
    print(f"  Greedy 推理动作: {action.numpy()}")
    
    # Actor-Critic 测试
    print("\n[Actor-Critic 测试]")
    ac_model = MahjongActorCritic()
    logits, value = ac_model(dummy_input, dummy_mask)
    print(f"  Logits shape: {logits.shape}")
    print(f"  Value shape: {value.shape}")
    print(f"  Value 样例: {value[0].item():.4f}")
    
    print("\n✅ 模型自检通过!")


class MahjongActorCritic(MahjongPolicyResNet):
    """
    麻将 Actor-Critic 网络 (PPO 专用)
    
    共享 ResNet 骨干，具备两个输出头：
    1. Actor (Policy Head): [B, 30] raw logits
    2. Critic (Value Head): [B, 1] scalar state value
    """
    
    def __init__(self, in_channels=16, num_actions=30, hidden_channels=256, num_res_blocks=5):
        super().__init__(in_channels, num_actions, hidden_channels, num_res_blocks)
        
        # 价值头 (Value Head/Critic)
        # 结构: Conv1d -> BN -> ReLU -> Flatten -> FC -> ReLU -> FC
        self.value_head = nn.Sequential(
            nn.Conv1d(hidden_channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 30, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)  # 输出状态价值 V(s)
        )
        
        # 再次初始化 (主要是新加的 value_head)
        self._init_weights()
        
    def forward(self, x, legal_mask=None):
        """
        Actor-Critic 前向传播
        
        Returns:
            logits: [B, 30] 动作分布
            value: [B, 1] 状态价值
        """
        # 共享特征提取
        features = self.input_conv(x)
        features = self.res_blocks(features)
        
        # 策略头
        logits = self.policy_head(features)
        if legal_mask is not None:
            logits = logits.masked_fill(legal_mask == 0, float('-inf'))
            
        # 价值头
        value = self.value_head(features)
        
        return logits, value
    
    def evaluate_actions(self, x, action, legal_mask=None):
        """
        用于 PPO 更新: 计算动作的 log_prob, 熵以及状态价值
        """
        logits, value = self.forward(x, legal_mask)
        probs = F.softmax(logits, dim=-1)
        
        # 避免数值不稳定性，对非法动作的概率做微小修正
        if legal_mask is not None:
            probs = probs * legal_mask
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
            
        dist = torch.distributions.Categorical(probs)
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return log_prob, value, entropy
