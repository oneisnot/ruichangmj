import os
import time
import sys
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from resnet_model import MahjongActorCritic
from ruichang_mj_env import RuichangMahjongEnv
# get_device: CUDA优先 (V100), CPU兜底
def get_device():
    if hasattr(get_device, '_cached'):
        return get_device._cached
        
    torch = __import__('torch')
    # 针对 V100 + 特殊 Torch 版本的环境兼容性修复
    torch.backends.cudnn.enabled = False 
    
    if torch.cuda.is_available():
        dev = __import__('torch').device('cuda:0')
        name = __import__('torch').cuda.get_device_name(0)
        log_message(f"🚀 CUDA GPU: {name}")
    else:
        dev = __import__('torch').device('cpu')
        log_message("⚠️  CUDA不可用, 降级CPU")
    get_device._cached = dev
    return dev

ACTOR_LR = 1e-5            # 核心：降低 Actor 学习率以进行精细手术后的微调
CRITIC_LR = 3e-4           
GAMMA = 0.99
GAE_LAMBDA = 0.95
EPS_CLIP = 0.2
K_EPOCHS = 4
ENT_START = 0.05           # 初始探索系数：适应新规则层
ENT_END = 0.005            # 最终收敛系数
ENT_DECAY_STEPS = 1000000 
TRAIN_STEPS = 1500000      
TRAIN_BATCH_SIZE = 1024    # 缩短更新周期，提升反馈频率
NUM_WORKERS = 20           # 留出 4 个超线程给系统和推理 Server 确保不占满 CPU
def log_message(msg):
    timestamp = time.strftime('%H:%M:%S')
    formatted_msg = f"[{timestamp}] {msg}"
    print(formatted_msg, flush=True)
    with open("training.log", "a", encoding="utf-8") as f:
        f.write(formatted_msg + "\n")
        f.flush()

def rollout_worker(worker_id, request_queue, response_queue, training_queue):
    log_message(f"Worker {worker_id} started.")
    try:
        env = RuichangMahjongEnv()
        while True:
            state = env.reset()
            if env.done:
                # 游戏在reset时已结束（如其他玩家起手胡牌），无有效步骤，跳过
                continue
            done = False
            trajectory = []
            while not done:
                mask = state[13]
                request_queue.put((worker_id, state, mask))
                action, log_prob, val = response_queue.get()
                next_state, reward, done, _ = env.step(action)
                trajectory.append({
                    'state': state, 'action': action, 'reward': reward,
                    'log_prob': log_prob, 'value': val, 'mask': mask, 'done': done
                })
                state = next_state
            training_queue.put(trajectory)
    except Exception as e:
        log_message(f"Worker {worker_id} CRASHED: {e}")
        traceback.print_exc()

def inference_server(request_queue, response_queues, model_weights_shared):
    device = get_device()
    log_message(f"Inference Server running on: {device}")
    model = MahjongActorCritic().to(device)
    model.eval()
    
    # 【修复点 1】：禁止推理机自行加载旧模型 (防止 14/16 通道维度撕裂崩溃)
    # 强制阻塞，等待 Trainer 完成“外科手术”并将新维度权重发过来
    log_message("Inference Server: 等待 Trainer 注入已完成手术的初始权重...")
    try:
        new_weights_np = model_weights_shared.get() # 阻塞式等待初始权重
        new_weights = {k: torch.from_numpy(v).to(device) for k, v in new_weights_np.items()}
        model.load_state_dict(new_weights)
        log_message("✅ Inference Server: 初始权重接收完毕，开启高频推理循环！")
    except Exception as e:
        log_message(f"❌ Inference Server 初始加载失败: {e}")
        return

    while True:
        if not model_weights_shared.empty():
            try:
                new_weights_np = model_weights_shared.get_nowait()
                new_weights = {k: torch.from_numpy(v).to(device) for k, v in new_weights_np.items()}
                model.load_state_dict(new_weights)
            except: pass

        batch_items = []
        for _ in range(32):    # 降低聚合批次至 32，消除 22 个 Worker 下的死锁风险
            try:
                timeout = 0.001 if not batch_items else 0.0
                batch_items.append(request_queue.get(timeout=timeout))
            except: break
        
        if not batch_items: continue
            
        worker_ids, states, masks = zip(*batch_items)
        states_t = torch.from_numpy(np.stack(states)).to(device).float()
        masks_t = torch.from_numpy(np.stack(masks)).to(device).float()
        
        with torch.no_grad():
            logits, values = model(states_t, legal_mask=masks_t)
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
            
        for i, wid in enumerate(worker_ids):
            response_queues[wid].put((actions[i].item(), log_probs[i].item(), values[i].item()))

def compute_gae(rewards, values, next_value, dones):
    returns, advantages = [], []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + GAMMA * next_value * (1 - dones[i]) - values[i]
        gae = delta + GAMMA * 0.95 * (1 - dones[i]) * gae
        next_value = values[i]
        returns.insert(0, gae + values[i])
        advantages.insert(0, gae)
    return returns, advantages

class PPOTrainer:
    def __init__(self, model_weights_shared):
        self.device = get_device()
        log_message(f"Trainer running on: {self.device}")
        self.model = MahjongActorCritic().to(self.device)
        self.model_weights_shared = model_weights_shared
        
        # 1. 学习率解耦 (Dual Learning Rate)
        # Backbone + PolicyHead = Actor (Slow)
        # ValueHead = Critic (Fast)
        policy_params = list(self.model.input_conv.parameters()) + \
                        list(self.model.res_blocks.parameters()) + \
                        list(self.model.policy_head.parameters())
        value_params = list(self.model.value_head.parameters())
        
        self.optimizer = optim.AdamW([
            {'params': policy_params, 'lr': ACTOR_LR},
            {'params': value_params, 'lr': CRITIC_LR}
        ], weight_decay=1e-4) # 增加微小正则化
        
        self.load_checkpoint()

    def _sync_weights(self):
        """将 GPU 训练权重同步到推理队列 (Queue 模型)"""
        # 必须转到 CPU 并转为 numpy，否则多进程无法反序列化 CUDA Tensor 或造成 Tensor/Numpy 混淆
        np_weights = {k: v.detach().cpu().numpy() for k, v in self.model.state_dict().items()}
        try:
            while not self.model_weights_shared.empty():
                self.model_weights_shared.get_nowait()
            self.model_weights_shared.put_nowait(np_weights)
        except: pass

    def load_sl_baseline_with_surgery(self, sl_weights_path):
        """核心：将 14 通道的 SL 模型权重外科手术式嫁接到 16 通道模型"""
        log_message(f"Trainer: Performing Network Surgery from {sl_weights_path}...")
        old_state_dict = torch.load(sl_weights_path, map_location="cpu", weights_only=False)
        if 'model_state_dict' in old_state_dict:
            old_state_dict = old_state_dict['model_state_dict']
            
        new_state_dict = self.model.state_dict()
        
        for name, param in old_state_dict.items():
            if name not in new_state_dict: continue
                
            if name == "input_conv.0.weight":
                # 处理第一层卷积核维度扩充 [256, 14, 3] -> [256, 16, 3]
                log_message(f"[Surgery] Grafting layer: {name} {param.shape} -> {new_state_dict[name].shape}")
                new_param = new_state_dict[name].clone()
                # 复制原有的 14 层特征逻辑
                new_param.data[:, :14, :] = param.data
                # 新增的 2 层 (庄家, 底分) 初始化为 0，防止逻辑突变
                torch.nn.init.zeros_(new_param.data[:, 14:, :])
                new_state_dict[name] = new_param
            else:
                # 其他层直接对齐加载
                new_state_dict[name] = param
                
        self.model.load_state_dict(new_state_dict)
        self._sync_weights()
        log_message("✅ SL 模型手术衔接完成。")

    def load_checkpoint(self):
        # 此时我们主动废弃旧的 ppo 存档，强制从手术后的 SL 开始
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sl_path = os.path.join(script_dir, "checkpoints/best_policy.pth")
        
        if os.path.exists(sl_path):
            self.load_sl_baseline_with_surgery(sl_path)
        else:
            log_message("Trainer: WARNING - No SL baseline found, starting from random weights.")

    def update(self, storage, total_steps):
        # 2. 动态熵衰减 (Entropy Decay)
        entropy_coef = max(ENT_END, ENT_START - (ENT_START - ENT_END) * (total_steps / ENT_DECAY_STEPS))
        
        states = torch.stack([torch.from_numpy(s['state']) for s in storage]).to(self.device).float()
        actions = torch.tensor([s['action'] for s in storage]).to(self.device).long()
        old_log_probs = torch.tensor([s['log_prob'] for s in storage]).to(self.device).float()
        returns = torch.tensor([s['return'] for s in storage]).to(self.device).float()
        advantages = torch.tensor([s['advantage'] for s in storage]).to(self.device).float()
        masks = torch.stack([torch.from_numpy(s['mask']) for s in storage]).to(self.device).float()
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        v_losses, p_losses = [], []
        for _ in range(K_EPOCHS):
            log_probs, values, dist_entropy = self.model.evaluate_actions(states, actions, masks)
            ratios = torch.exp(log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-EPS_CLIP, 1+EPS_CLIP) * advantages
            
            p_loss = -torch.min(surr1, surr2).mean()
            # 使用 Huber Loss (SmoothL1) 替代 MSE，增强对 100 炮奖励冲击的鲁棒性
            v_loss = F.smooth_l1_loss(values.squeeze(), returns)
            e_loss = -entropy_coef * dist_entropy.mean()
            
            loss = p_loss + v_loss + e_loss
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            v_losses.append(v_loss.item())
            p_losses.append(p_loss.item())
            
        self._sync_weights()
        return np.mean(p_losses), np.mean(v_losses), entropy_coef

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(script_dir, "training.log")
    
    # 全天候参数化训练：开启全新日志
    if os.path.exists(log_file):
        os.rename(log_file, log_file + f".{int(time.time())}.bak")
        
    runs_dir = os.path.join(script_dir, f"runs/ppo_conditioned_{int(time.time())}")
    writer = SummaryWriter(log_dir=runs_dir)
    
    # 【修复点 2】：无论 Linux 还是 Windows，只要用 CUDA，坚决使用 spawn！
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
        
    request_queue = mp.Queue(maxsize=1024)
    response_queues = [mp.Queue(maxsize=1) for _ in range(NUM_WORKERS)]
    training_queue = mp.Queue(maxsize=200)
    model_weights_shared = mp.Queue(maxsize=1)
    
    inf_proc = mp.Process(target=inference_server, args=(request_queue, response_queues, model_weights_shared))
    inf_proc.start()
    time.sleep(15) 
    
    worker_procs = []
    for i in range(NUM_WORKERS):
        p = mp.Process(target=rollout_worker, args=(i, request_queue, response_queues[i], training_queue))
        p.start()
        worker_procs.append(p)
        time.sleep(0.5) 
        
    trainer = PPOTrainer(model_weights_shared)
    total_steps = 0
    log_message("PPO Conditioned Training (Domain Randomization) Start.")
    
    all_storage = []
    recent_ep_rewards = []
    recent_ep_lengths = []
    try:
        while True:
            trajectory = training_queue.get()
            ep_reward = sum([s['reward'] for s in trajectory])
            recent_ep_rewards.append(ep_reward)
            recent_ep_lengths.append(len(trajectory))

            returns, advantages = compute_gae(
                [s['reward'] for s in trajectory],
                [s['value'] for s in trajectory],
                0, [s['done'] for s in trajectory]
            )
            for i, step_data in enumerate(trajectory):
                step_data['return'], step_data['advantage'] = returns[i], advantages[i]
                all_storage.append(step_data)
            
            total_steps += len(trajectory)
            if len(all_storage) >= TRAIN_BATCH_SIZE:
                p_loss, v_loss, ent = trainer.update(all_storage, total_steps)
                avg_ep_reward = np.mean(recent_ep_rewards[-100:]) if recent_ep_rewards else 0.0
                avg_ep_len = np.mean(recent_ep_lengths[-100:]) if recent_ep_lengths else 0.0
                
                log_message(f"STEPS: {total_steps} | P_LOSS: {p_loss:.4f} | V_LOSS: {v_loss:.4f} | ENT: {ent:.4f} | REWARD: {avg_ep_reward:.2f}")
                
                writer.add_scalar("Loss/Policy", p_loss, total_steps)
                writer.add_scalar("Loss/Value", v_loss, total_steps)
                writer.add_scalar("Stats/Entropy", ent, total_steps)
                writer.add_scalar("Reward/Episode", avg_ep_reward, total_steps)
                writer.add_scalar("Stats/EpisodeLength", avg_ep_len, total_steps)
                writer.flush() # 强制刷新 TensorBoard
                
                all_storage = []
                
                if total_steps % 50000 < TRAIN_BATCH_SIZE:
                    save_path = os.path.join(script_dir, "checkpoints/ppo_elite_v1.pth")
                    save_dict = {
                        'model_state_dict': trainer.model.state_dict(),
                        'total_steps': total_steps
                    }
                    torch.save(save_dict, save_path)
    except KeyboardInterrupt: pass
    finally:
        inf_proc.terminate()
        for p in worker_procs: p.terminate()
        writer.close()

if __name__ == "__main__":
    main()
