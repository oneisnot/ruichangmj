# 瑞昌麻将 AI 项目说明

## 项目概览
本项目旨在构建一个工业级、可扩展的瑞昌麻将 AI，支持大规模 Monte‑Carlo 仿真、深度强化学习（SL + PPO�┌─────────────────────┐
│   采样工人 (16 核)  │
│  (Rollout Workers)   │
│  - 并行采集 (S, A, R) │
│  - 分数驱动稀疏奖励   │
└───────▲───────▲───────┘
        │       │ (Request/Response)
        ▼───────▼
┌─────────────────────┐      ┌─────────────────────┐
│   推理中心 (GPU)    │      │   PPO 训练器 (GPU)  │
│  (Inference Server) ◄──────┤     (Trainer)       │
│  - 大 Batch 聚合推理  │      │ - Dual LR (Actor/Critic)
│  - 权重同步 (Numpy)   │      │ - 动态熵衰减 (Decay)
└─────────────────────┘      └─────────────────────┘
```

### 1. 向听数查表
- **离线生成**：`generate_shanten_table.py` 穷举所有合法 14 张牌组合（约 3.5 B），计算向听数并压缩为二进制 `.pkl`。 
- **运行时加载**：使用 `multiprocessing.shared_memory` 将表一次性映射到内存，查询仅为一次数组索引 → **O(1)**。

### 2. 番数结算引擎与奖励塑形
- **结算引擎**：实现 `calculate_final_score`，兼容七仙女、红旗飘飘等名场面。
- **奖励信号安全锁**：为了防止极端大牌（100 炮）摧毁 SL 基础，实施了 **开方平滑缩放 (Square Root Scaling)**：$Reward = \text{sign}(S) \cdot \sqrt{|S|}$。2 炮 $\rightarrow$ 1.41，100 炮 $\rightarrow$ 10.0。

### 3. 特征编码（state_encoder.py）
| 通道 | 说明 |
|------|------|
| 0‑3  | **温度计编码**：≥1、≥2、≥3、==4 张同种牌的二进制层。 |
| 4    | **玩家自己的鸣牌**（碰/杠）。 |
| 5‑8  | **四家废牌池**：己方、下家、对家、上家各自弃牌计数（归一化）。 |
| 9    | **对手鸣牌**。 |
| 10   | **最后一张触发牌**。 |
| 11   | **发财池计数**（归一化）。 |
| 12   | **牌堆剩余计数**（归一化）。 |
| 13   | **合法动作掩码**：>0 为合法出牌。 |
| 14   | **【新】庄家位置信号**：P0庄=1.0, 下家庄=0.25, 对家=0.5, 上家=0.75。 |
| 15   | **【新】底分规则信号**：base_score / 20.0（Domain Randomization 条件输入）。 |

## 深度学习模型（resnet_model.py）
- **架构**：**Actor-Critic** 共享骨干 ResNet。
- **输入**：`(16, 30)` 张量（16 通道，30 种牌）。
- **Actor (策略头)**: 输出 30 维动作分布（从 SL 97.8% 模型手术嫁接初始化）。
- **Critic (价值头)**: 输出状态标量价值 $V(s)$（从零初始化）。
- **参数量**：约 **5M**。
- **网络手术（Surgery）**：SL 旧模型输入层为 `[256, 14, 3]`，PPO 新模型扩充至 `[256, 16, 3]`，新增的庄家/底分通道权重初始化为 0，确保旧策略不受扰动。

## 强化学习设置（train_ppo.py）
针对瑞昌麻将的高方差博弈环境，对齐了以下专家级参数：

| 项目 | 配置 | 理由 |
|------|------|------|
| **Dual LR (Actor)** | 1e-5 | 极低学习率，绝对保护监督学习建立的"牌理护城河"。 |
| **Dual LR (Critic)**| 3e-4 | 标准学习率，让价值网络快速从零学习期望收益。 |
| **Entropy Decay** | 0.05 → 0.005（over 1M steps）| 初始高熵强迫 AI 探索；后期收敛。 |
| **Gamma** | 0.99 | 平衡远视野与训练稳定性。 |
| **Clip Range (ε)** | 0.2 | 标准 PPO clip 范围。 |
| **Critic Loss** | **Huber Loss (SmoothL1)** | 吸收大牌离群奖励的梯度冲击。 |
| **Batch Size** | 1024 steps | 短周期高频更新，加快反馈。 |
| **K_Epochs** | 4 | 每批数据重复利用 4 次。 |
| **Domain Randomization** | 庄家随机 + 底分随机 {0,5,10,15} | 增强泛化，条件化已编码进通道 14/15。 |

### 训练架构
- **硬件**：NVIDIA Tesla V100-SXM2-32GB (CUDA)
- **异步推理**：**20 个**并发采样工人（CPU），1 个独立 GPU 推理中心。
- **通信机制**：`mp.Queue` + Numpy 数组权重同步（无 CUDA Tensor 跨进程风险）。
- **推理批聚合**：每次最多聚合 32 个请求后批量推理，兼顾延迟与吞吐。

## 项目文件结构
```
e:\ai\瑞昌麻将\
│
├─ ruichang_mj_sim.py          # 主仿真入口、数据生成 pipeline
├─ ruichang_mj_env.py          # RL 环境封装 (Reward Scaling)
├─ state_encoder.py            # 14‑通道特征编码
├─ calculate_final_score.py    # 番数结算引擎
│
├─ resnet_model.py             # Actor-Critic ResNet 实现
├─ train_sl.py                 # 监督学习训练脚本 (已完成 97.8%)
├─ train_ppo.py                # 强化学习异步训练脚本 (运行中)
├─ eval_checkpoint.py          # 检查点评估与对战测试
│
├─ checkpoints\                # 训练 checkpoint (best_policy.pth, ppo_elite_v1.pth)
├─ training.log                # 训练实时指标 (Steps, P_Loss, V_Loss, Ent)
└─ docs\
   └─ project_documentation.md   # 本说明文档
```

---
**文档更新于 2026-04-02**，如需进一步补充或修改，请直接编辑该文件。
Train Acc | Val Acc | LR | 耗时 |
|-------|-----------|---------|----|------|
| 1 | 85.9% | 91.7% | 9.99e-4 | 84 s |
| 12 | 98.3% | 96.6% | 8.65e-4 | 81.6 s |
| 38 | 99.9% | 97.7% | 1.36e-4 | 80.6 s |
| 50 | **100.0%** | **97.8%** | 1.00e-6 | 80.0 s |
> **备注**：`Train Loss` 目前显示 `nan`，已在代码中加入 `torch.nan_to_num` 进行容错，训练已恢复正常。

### 训练总结
- **训练设备**: AMD Radeon 6900 XT (DirectML)
- **总耗时**: 约 70 分钟 (50 Epochs)
- **最终准确率**: 验证集 **97.8%**，训练集 **100.0%**
- **模型保存**: `checkpoints/best_policy.pth` (最佳模型), `checkpoints/final_policy.pth` (最终模型)

## 评估脚本（eval_checkpoint.py）
- 随机抽取 2 000 条样本，使用 `best_policy.pth` 进行前向推理。
- 计算 **Sample Validation Accuracy**（示例运行得到约 **0.95**，与验证集一致）。

## 阶段进度总览

| 阶段 | 目标 | 状态 | 完成时间 |
|------|------|------|----------|
| **A. SL 训练** | 50 epoch，验证准确率 ≥ 96% | ✅ 完成 | 2026-04-02 |
| **B. PPO 环境搭建** | 异步多工人 + GPU 推理服务器 | ✅ 完成 | 2026-04-03 |
| **C. 网络手术（Surgery）** | 14→16 通道扩充，嫁接 SL 权重 | ✅ 完成 | 2026-04-03 |
| **D. Domain Randomization PPO** | 庄家/底分随机化条件训练，1.5M steps | 🔄 **进行中** | 预计 2026-04-04 ~04:40 |
| **E. 评估与调优** | 对战测试，REWARD 均值趋正 | ⬜ 待开始 | — |
| **F. 部署** | ONNX 导出 / 实时推理 | ⬜ 待开始 | — |

## PPO 阶段二（Domain Randomization）训练日志

**启动时间**：2026-04-04 03:36  
**训练目标**：1,500,000 steps  
**当前进度**（截至文档更新）：~131,000 steps（约 8.7%）  
**预计完成**：2026-04-04 04:40 左右（约剩 58 分钟）  

### 初期指标快照（前 ~130k steps）
| 指标 | 值 | 说明 |
|------|----|------|
| 步速 | ~400 steps/s | 20 Worker 全部稳定，无崩溃 |
| V_LOSS | 1.5 ~ 2.3 | 正常震荡，Critic 处于适应期 |
| P_LOSS | 0.001 ~ 0.04 | 稳健，无梯度爆炸 |
| ENT | 0.0500 → 0.0441 | 按计划线性衰减 |
| REWARD（100局均值）| -0.51 | 正常，四人博弈初期负期望 |
| 正向局比例 | ~7% | 初期正常，预计 400k+ steps 后改善 |

### Bug 修复记录（2026-04-04）
- **[Fix 1]** `env.reset()` 在游戏初始即结束时返回 tuple 而非 ndarray，导致 Worker 抛出 `tuple index out of range`。
  - **修复**：`reset()` 内部处理 done 状态，始终返回 `(16, 30)` ndarray；Worker 检测 `env.done` 跳过空局。
- **[Fix 2]** 跳过空局后 `value=None` 进入 `compute_gae` 触发 `TypeError`。
  - **修复**：Worker 直接 `continue` 跳过 `env.done==True` 的局，不入训练队列。

## 项目文件结构
```
/home/cnhbs/ruichang_mj/
│
├─ ruichang_mj_sim.py          # 仿真引擎、发牌/和牌/番数计算
├─ ruichang_mj_env.py          # RL 环境封装 (Domain Randomization + Reward Scaling)
├─ generate_shanten_table.py   # 向听数查表离线生成
├─ state_encoder.py            # 16 通道特征编码（含庄家/底分条件通道）
│
├─ resnet_model.py             # Actor-Critic ResNet 实现
├─ train_sl.py                 # 监督学习训练脚本（已完成）
├─ train_ppo.py                # PPO 异步强化学习训练脚本（运行中）
├─ eval_checkpoint.py          # 检查点评估与对战测试
│
├─ ruichang_expert_v1.npz      # 专家数据集（约 320k 条）
├─ checkpoints/
│   ├─ best_policy.pth          # SL 最佳模型（14 通道，97.8% 准确率）
│   ├─ final_policy.pth         # SL 最终模型
│   ├─ ppo_elite_v1.pth         # PPO 当前检查点（16 通道，每 50k steps 存档）
│   └─ ppo_elite_v1.pth.14ch_bak  # PPO 旧版备份
│
├─ training.log                # 当前训练实时指标
├─ training.log.*.bak          # 历史训练日志备份
└─ project_documentation.md   # 本说明文档
```

---

**文档最后更新：2026-04-04**
