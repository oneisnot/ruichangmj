import os
import time
import torch
import numpy as np
from collections import Counter
from resnet_model import MahjongActorCritic
from ruichang_mj_env import RuichangMahjongEnv
import ruichang_mj_sim as sim

def evaluate_ppo(num_games=100, checkpoint_name="ppo_elite_v1.pth", greedy=True):
    print(f"[{time.strftime('%H:%M:%S')}] 启动 PPO 模型性能评估流水线...")
    print(f"目标局数: {num_games} | 策略类型: {'Greedy (贪婪)' if greedy else 'Stochastic (随机采样)'}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(script_dir, "checkpoints", checkpoint_name)
    
    if not os.path.exists(ckpt_path):
        print(f"❌ 找不到权重文件: {ckpt_path}")
        return

    # 1. 环境与模型初始化
    env = RuichangMahjongEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MahjongActorCritic().to(device)
    
    # 2. 加载权重 (兼容 dict 和 state_dict 两种格式)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"✅ 已加载 PPO 训练权重 (Step: {ckpt.get('total_steps', 'Unknown')})")
    else:
        model.load_state_dict(ckpt)
        print(f"✅ 已加载模型 state_dict")
    model.eval()

    # 3. 统计指标初始化
    stats = {
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "total_reward": 0,
        "win_patterns": Counter(),
        "ep_lengths": [],
        "by_base_score": {
            0: {"wins": 0, "draws": 0, "losses": 0, "rewards": []},
            5: {"wins": 0, "draws": 0, "losses": 0, "rewards": []},
            10: {"wins": 0, "draws": 0, "losses": 0, "rewards": []},
            15: {"wins": 0, "draws": 0, "losses": 0, "rewards": []}
        }
    }

    start_time = time.time()

    # 4. 模拟循环
    for g in range(num_games):
        obs = env.reset()
        done = False
        ep_reward = 0
        turns = 0
        base_score = env.current_base_score  # 记录这局的base_score
        
        while not done:
            turns += 1
            # 准备输入张量
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
            # 合法动作掩码从第13通道提取
            legal_mask = torch.from_numpy(obs[13]).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits, _ = model(obs_t, legal_mask=legal_mask)
                if greedy:
                    action = torch.argmax(logits, dim=-1).item()
                else:
                    probs = torch.softmax(logits, dim=-1)
                    action = torch.multinomial(probs, 1).item()
            
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
            
        # 结果记录
        stats["total_reward"] += ep_reward
        stats["ep_lengths"].append(turns)
        stats["by_base_score"][base_score]["rewards"].append(ep_reward)
        
        # 判定赢家 (Player 0 是 AI)
        # 注意: env._finalize_game 返回的 reward 机制
        # win: reward > 0, draw: reward == -0.5, loss: reward < -0.5 OR reward == 0 (别人点炮但AI没输)
        # 我们通过直接检查 env 内部状态来更精确判定
        winner = -1
        # 查找这局谁胡了
        for i, p in enumerate(env.players):
            if sim.calculate_shanten_accurate(p["hand"], p["melds"]) == -1:
                winner = i
                break
        
        if winner == 0:
            stats["wins"] += 1
            stats["by_base_score"][base_score]["wins"] += 1
            # 记录胡牌番种
            p = env.players[0]
            # 为了获取番种，我们模拟一下最后一次 score 计算
            # 注意: 我们需要 win_tile，这里简单用 last_tile
            _, patterns = sim.calculate_final_score(p, env.last_tile, True, {})
            for name, _ in patterns:
                stats["win_patterns"][name] += 1
        elif winner == -1:
            stats["draws"] += 1
            stats["by_base_score"][base_score]["draws"] += 1
        else:
            stats["losses"] += 1
            stats["by_base_score"][base_score]["losses"] += 1
            
        if (g + 1) % 20 == 0:
            print(f"  Progress: {g+1}/{num_games} | Current WinRate: {stats['wins']/(g+1)*100:.1f}%")

    end_time = time.time()
    
    # 5. 输出汇总报告
    duration = end_time - start_time
    win_rate = (stats["wins"] / num_games) * 100
    draw_rate = (stats["draws"] / num_games) * 100
    loss_rate = (stats["losses"] / num_games) * 100
    avg_reward = stats["total_reward"] / num_games
    avg_len = np.mean(stats["ep_lengths"])

    print("\n" + "="*50)
    print("🏆 PPO 模型实战评估报告")
    print("="*50)
    print(f"评估局数: {num_games}")
    print(f"总耗时: {duration:.2f}s ({num_games/duration:.2f} games/s)")
    print(f"平均奖励: {avg_reward:.4f}")
    print(f"平均局长: {avg_len:.1f} 巡")
    print("-" * 30)
    print(f"胜率 (Win Rate): {win_rate:.1f}%")
    print(f"流局率 (Draw Rate): {draw_rate:.1f}%")
    print(f"败率 (Loss Rate): {loss_rate:.1f}%")
    print("-" * 30)
    
    # 【新增】底分规则分化分析
    print("\n📊 AI 在不同 Base Score 下的表现分化：")
    print("-" * 50)
    for bs in [0, 5, 10, 15]:
        data = stats["by_base_score"][bs]
        total = data["wins"] + data["draws"] + data["losses"]
        if total > 0:
            wr = data["wins"] / total * 100
            dr = data["draws"] / total * 100
            lr = data["losses"] / total * 100
            avg_r = np.mean(data["rewards"]) if data["rewards"] else 0
            print(f"Base Score = {bs:2d}  | 胜率: {wr:5.1f}% | 流局: {dr:5.1f}% | 败率: {lr:5.1f}% | 平均Reward: {avg_r:7.3f}")
    print("-" * 50)
    print("💡 观察：是否出现了在 Base=0 时左右安全、Base=15 时激进进攻的打法分化？")
    
    print("\n" + "-" * 30)
    print("🔥 胡牌番种统计 (前5名):")
    for name, count in stats["win_patterns"].most_common(5):
        print(f"  - {name}: {count} 次 ({count/stats['wins']*100:.1f}% of wins)")
    print("="*50)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=100, help="评估局数")
    parser.add_argument("--ckpt", type=str, default="ppo_elite_v1.pth", help="权重文件名")
    parser.add_argument("--stochastic", action="store_true", help="使用随机采样策略")
    args = parser.parse_args()
    
    evaluate_ppo(num_games=args.games, checkpoint_name=args.ckpt, greedy=not args.stochastic)
