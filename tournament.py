import os
import time
import random  # 修复：移至全局作用域
import argparse
import torch
import numpy as np
from collections import Counter

from resnet_model import MahjongActorCritic
import ruichang_mj_sim as sim
from state_encoder import encode_state

def simulate_tournament_game(model, device):
    """单局锦标赛：1 个 RL 宗师 (Player 0) VS 3 个启发式查表专家"""
    deck = sim.create_deck()
    players = [{"hand": [], "melds": [], "fa_count": 0, "is_tenpai": False} for _ in range(4)]
    discards_by_player = [[], [], [], []]
    
    dealer_idx = random.randint(0, 3)
    current_player = dealer_idx
    base_score = random.choice([0, 5, 10, 15])
    last_tile = None
    turns = 0
    
    # 发牌与补花
    for _ in range(13):
        for p in players: p["hand"].append(deck.pop())
    for p in players:
        while sim.FA in p["hand"]:
            p["hand"].remove(sim.FA)
            p["fa_count"] += 1
            if deck: p["hand"].append(deck.pop(0))

    while deck:
        turns += 1
        p = players[current_player]
        
        # 摸牌
        drawn_tile = deck.pop()
        while drawn_tile == sim.FA and deck:
            p["fa_count"] += 1
            drawn_tile = deck.pop(0)
            
        p["hand"].append(drawn_tile)
        last_tile = drawn_tile
        
        # 查自摸
        if sim.calculate_shanten_accurate(p["hand"], p["melds"]) == -1:
            score, patterns = sim.calculate_final_score(p, drawn_tile, True, {})
            if score >= 2:
                return {"winner": current_player, "is_zimo": True, "score": score, 
                        "patterns": patterns, "turns": turns, "base_score": base_score}

        # 弃牌决策
        discarded_tile = None
        if current_player == 0:
            # RL 宗师决策
            tot_fa = sum(pl["fa_count"] for pl in players)
            opp_melds = []
            for i in range(1, 4): opp_melds.extend(players[i]["melds"])
            
            # 视角依然是 Player 0，相对庄家位置就是 dealer_idx
            obs = encode_state(p["hand"], p["melds"], discards_by_player, opp_melds, 
                               tot_fa, len(deck), last_tile, dealer_idx, base_score)
            
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
            mask_t = torch.from_numpy(obs[13]).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits, _ = model(obs_t, legal_mask=mask_t)
                discarded_tile = torch.argmax(logits, dim=-1).item() # 锦标赛使用绝对贪婪策略
                
            if discarded_tile not in p["hand"]: discarded_tile = p["hand"][0]
        else:
            # 查表专家决策
            discarded_tile = sim.get_best_discard_smart(p["hand"], p["melds"])
            
        p["hand"].remove(discarded_tile)

        # 查点炮
        hu_triggered = False
        for i in range(1, 4):
            target_idx = (current_player + i) % 4
            tp = players[target_idx]
            tp["hand"].append(discarded_tile)
            if sim.calculate_shanten_accurate(tp["hand"], tp["melds"]) == -1:
                score, patterns = sim.calculate_final_score(tp, discarded_tile, False, {})
                if score >= 2:
                    return {"winner": target_idx, "is_zimo": False, "score": score, 
                            "patterns": patterns, "turns": turns, "base_score": base_score, "loser": current_player}
            tp["hand"].remove(discarded_tile)

        # 查碰杠 (简化逻辑：仅查表专家会碰杠，测验RL的防守)
        action_taken = False
        for i in range(1, 4):
            target_idx = (current_player + i) % 4
            if target_idx == 0: continue # RL在测试脚本中保持不碰不杠(测纯出牌调度能力)
            tp = players[target_idx]
            count = tp["hand"].count(discarded_tile)
            
            if count >= 3:
                tp["hand"] = [t for t in tp["hand"] if t != discarded_tile]
                tp["melds"].append(("KONG", discarded_tile))
                if deck:
                    bonus = deck.pop(0)
                    while bonus == sim.FA and deck:
                        tp["fa_count"] += 1
                        if not deck: break
                        bonus = deck.pop(0)
                    tp["hand"].append(bonus)
                current_player = target_idx
                action_taken = True
                break
            elif count == 2 and sim.should_pong(tp["hand"], discarded_tile):
                tp["hand"] = [t for t in tp["hand"] if t != discarded_tile]
                tp["melds"].append(("PONG", discarded_tile))
                current_player = target_idx
                action_taken = True
                break
                
        if not action_taken:
            discards_by_player[current_player].append(discarded_tile)
            current_player = (current_player + 1) % 4
            
    return {"winner": -1, "is_zimo": False, "score": 0, "patterns": [], "turns": turns, "base_score": base_score}

def run_tournament(games=1000, ckpt_name="ppo_elite_v1.pth"):
    print(f"[{time.strftime('%H:%M:%S')}] 🎺 瑞昌麻将紫禁之巅：Self-Play 宗师 VS 查表专家群")
    print(f"正在准备 {games} 局标准锦标赛... 目标模型: {ckpt_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 如果 detected GPU 的计算能力低于构建的 PyTorch 支持范围，回退到 CPU 避免运行时引擎错误
    if device.type == "cuda":
        try:
            cc = torch.cuda.get_device_capability()
            if (cc[0] * 10 + cc[1]) < 75:
                print(f"⚠️ 检测到 GPU 计算能力 {cc}，低于推荐的 7.5，回退使用 CPU 以避免不兼容")
                device = torch.device("cpu")
        except Exception:
            # 若查询能力失败则保留原始 device
            pass

    model = MahjongActorCritic().to(device)
    
    ckpt_path = os.path.join(os.path.dirname(__file__), "checkpoints", ckpt_name)
    if not os.path.exists(ckpt_path):
        print(f"❌ 致命错误：找不到模型文件 {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    print("✅ 宗师模型加载完毕，骨骼清奇，杀气内敛！\n")

    stats = {
        "rl_wins": 0, "expert_wins": 0, "draws": 0, "rl_dianpao": 0,
        "base_score_stats": {bs: {"plays": 0, "wins": 0, "draws": 0, "dianpao": 0} for bs in [0, 5, 10, 15]},
        "win_patterns": Counter(),
        "turns": []
    }

    start_time = time.time()
    for g in range(games):
        res = simulate_tournament_game(model, device)
        bs = res["base_score"]
        stats["base_score_stats"][bs]["plays"] += 1
        stats["turns"].append(res["turns"])
        
        if res["winner"] == 0:
            stats["rl_wins"] += 1
            stats["base_score_stats"][bs]["wins"] += 1
            for pat, _ in res["patterns"]: stats["win_patterns"][pat] += 1
        elif res["winner"] == -1:
            stats["draws"] += 1
            stats["base_score_stats"][bs]["draws"] += 1
        else:
            stats["expert_wins"] += 1
            if res.get("loser") == 0:
                stats["rl_dianpao"] += 1
                stats["base_score_stats"][bs]["dianpao"] += 1
                
        if (g + 1) % 100 == 0:
            print(f"  ⚔️ 激战中... {g+1}/{games} 局 | 宗师当前胜率: {stats['rl_wins']/(g+1)*100:.1f}%")

    duration = time.time() - start_time
    
    print("\n" + "★"*50)
    print("👑 瑞昌麻将 AI 巅峰对决 - 终极战报")
    print("★"*50)
    print(f"比赛规模: {games} 局 | 耗时: {duration:.1f} 秒 ({games/duration:.1f} 局/秒)")
    print(f"平均回合: {np.mean(stats['turns']):.1f} 巡")
    print("-" * 50)
    print(f"🥇 宗师胜率 (Win Rate)  : {stats['rl_wins']/games*100:.1f}%")
    print(f"🤝 流局率   (Draw Rate) : {stats['draws']/games*100:.1f}%")
    print(f"💀 点炮率   (Feed Rate) : {stats['rl_dianpao']/games*100:.1f}%  <-- (极其关键的生存指标)")
    print("-" * 50)
    
    print("📊 宗师在不同底分局(Base Score)的统治力：")
    for bs in [0, 5, 10, 15]:
        b_st = stats["base_score_stats"][bs]
        plays = b_st["plays"]
        if plays == 0: continue
        w_r = b_st["wins"] / plays * 100
        d_r = b_st["draws"] / plays * 100
        dp_r = b_st["dianpao"] / plays * 100
        print(f"  底分 {bs:2d} | 胜率: {w_r:4.1f}% | 流局: {d_r:4.1f}% | 点炮: {dp_r:4.1f}%")
        
    print("-" * 50)
    print("🔥 宗师绝技谱 (胡牌番种展示):")
    for name, count in stats["win_patterns"].most_common(6):
        print(f"  🀄 {name}: {count} 次 ({count/stats['rl_wins']*100:.1f}% of wins)")
    print("★"*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="瑞昌麻将紫禁之巅战报生成器")
    parser.add_argument("--games", type=int, default=1000, help="要模拟的锦标赛局数")
    parser.add_argument("--ckpt", type=str, default="ppo_elite_v1.pth", help="在 checkpoints 目录下的模型名称")
    args = parser.parse_args()
    
    run_tournament(games=args.games, ckpt_name=args.ckpt)