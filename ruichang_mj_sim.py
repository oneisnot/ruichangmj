import random
import multiprocessing
import time
from collections import Counter
import os
import pickle
import numpy as np
from state_encoder import encode_state

# --- 全局常量定义 ---
ZHONG = 27
BAI = 28
FA = 29

# --- 预加载离线向听数查表 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(SCRIPT_DIR, "ruichang_mj_shanten_cache.pkl")
SUIT_TABLE = {}
HONOR_TABLE = {}
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "rb") as f:
        cache_data = pickle.load(f)
        SUIT_TABLE = cache_data["suit_table"]
        HONOR_TABLE = cache_data["honor_table"]

def create_deck():
    deck = []
    for i in range(30): deck.extend([i] * 4)
    random.shuffle(deck)
    return deck

def evaluate_tile_value(tile, hand):
    if tile == FA: return 1000 
    count = hand.count(tile)
    if count >= 2: return 100 
    neighbors = 0
    if tile < ZHONG:
        if tile % 9 > 0 and (tile - 1) in hand: neighbors += 1
        if tile % 9 < 8 and (tile + 1) in hand: neighbors += 1
        if tile % 9 > 1 and (tile - 2) in hand: neighbors += 1
        if tile % 9 < 7 and (tile + 2) in hand: neighbors += 1
    if tile in (ZHONG, BAI): return 15
    return neighbors * 10

def should_pong(hand, discarded_tile):
    counts = Counter(hand)
    pairs = [t for t, c in counts.items() if c >= 2]
    if len(pairs) >= 4: return False
    if len(pairs) == 1 and discarded_tile in pairs: return False
    if discarded_tile < ZHONG:
        if discarded_tile % 9 > 0 and (discarded_tile - 1) in hand: return False
        if discarded_tile % 9 < 8 and (discarded_tile + 1) in hand: return False
    return True

# ==========================================
# 🛑 核心修复 1：严格的胡牌验证器 (拦截 DP 假胡)
# ==========================================
def check_melds(hand_remainder):
    if not hand_remainder: return True
    hand_remainder.sort()
    first = hand_remainder[0]
    count = hand_remainder.count(first)
    
    # 尝试作为刻子
    if count >= 3:
        h_copy = hand_remainder[:]
        h_copy.remove(first); h_copy.remove(first); h_copy.remove(first)
        if check_melds(h_copy): return True
        
    # 尝试作为顺子
    if first < ZHONG and first % 9 <= 6:
        if (first + 1) in hand_remainder and (first + 2) in hand_remainder:
            h_copy = hand_remainder[:]
            h_copy.remove(first); h_copy.remove(first + 1); h_copy.remove(first + 2)
            if check_melds(h_copy): return True
    return False

def check_standard_hu(hand):
    if not hand: return True
    counts = Counter(hand)
    # 寻找将牌
    for t, c in counts.items():
        if c >= 2:
            h_copy = hand[:]
            h_copy.remove(t); h_copy.remove(t)
            if check_melds(h_copy): return True
    return False

def is_seven_pairs(hand):
    if len(hand) != 14: return False
    counts = Counter(hand)
    # 修复七对判定：4张相同的算 2 个对子
    num_pairs = sum(c // 2 for c in counts.values())
    return num_pairs == 7

def calculate_shanten_accurate(hand, melds):
    meld_count = len(melds)
    counts = Counter(hand)
    num_pairs = sum(c // 2 for c in counts.values()) # 修复 4张同牌算1对的 Bug
    shanten_7 = 6 - num_pairs
    if meld_count > 0: shanten_7 = 100 # 有副露不能胡七对
    
    # ... DP 查表逻辑保持原样 ...
    h_arr = [0]*30
    for t in hand: h_arr[t] += 1
    w_sum = sum(h_arr[i] * (5**i) for i in range(9))
    t_sum = sum(h_arr[9+i] * (5**i) for i in range(9))
    b_sum = sum(h_arr[18+i] * (5**i) for i in range(9))
    h_sum = sum(h_arr[27+i] * (5**i) for i in range(3))

    w_res = SUIT_TABLE.get(w_sum, (([-1]*5), ([-1]*5)))
    t_res = SUIT_TABLE.get(t_sum, (([-1]*5), ([-1]*5)))
    b_res = SUIT_TABLE.get(b_sum, (([-1]*5), ([-1]*5)))
    h_res = HONOR_TABLE.get(h_sum, (([-1]*5), ([-1]*5)))
    
    def merge(res1, res2):
        out_no, out_pair = [-1]*5, [-1]*5
        rn1, rp1 = res1; rn2, rp2 = res2
        for g1 in range(5):
            t1_no, t1_p = rn1[g1], rp1[g1]
            if t1_no == -1 and t1_p == -1: continue
            for g2 in range(5 - g1):
                t2_no, t2_p = rn2[g2], rp2[g2]
                if t2_no == -1 and t2_p == -1: continue
                if t1_no != -1 and t2_no != -1: out_no[g1+g2] = max(out_no[g1+g2], t1_no + t2_no)
                if t1_p != -1 and t2_no != -1: out_pair[g1+g2] = max(out_pair[g1+g2], t1_p + t2_no)
                if t1_no != -1 and t2_p != -1: out_pair[g1+g2] = max(out_pair[g1+g2], t1_no + t2_p)
        for g in range(5):
            out_no[g] = min(out_no[g], 4-g)
            out_pair[g] = min(out_pair[g], 4-g)
        return (out_no, out_pair)
        
    m1 = merge(w_res, t_res)
    m2 = merge(m1, b_res)
    m3 = merge(m2, h_res)
    
    min_s = 8
    fn, fp = m3
    for g in range(5):
        if fn[g] != -1: min_s = min(min_s, 8 - 2*(g + meld_count) - fn[g])
        if fp[g] != -1: min_s = min(min_s, 8 - 2*(g + meld_count) - fp[g] - 1)
            
    s_num = min(shanten_7, min_s)
    
    # 🛑 【终极拦截器】：如果查表说胡了，强制进行严格规则校验！防 78899 假胡！
    if s_num <= -1:
        if is_seven_pairs(hand): return -1
        if len(hand) % 3 == 2 and check_standard_hu(hand): return -1
        return 0 # 假胡，退回向听数 0
        
    return s_num

def get_best_discard_smart(hand, melds):
    unique_tiles = list(set(hand))
    best_discard = unique_tiles[0]
    min_s = 100
    lowest_base_val = float('inf') 
    for tile in unique_tiles:
        hand.remove(tile)
        s_num = calculate_shanten_accurate(hand, melds) 
        hand.append(tile)
        if s_num < min_s:
            min_s = s_num; best_discard = tile; lowest_base_val = evaluate_tile_value(tile, hand)
        elif s_num == min_s:
            base_val = evaluate_tile_value(tile, hand)
            if base_val < lowest_base_val:
                lowest_base_val = base_val; best_discard = tile
    return best_discard

# ==========================================
# 🛑 核心修复 2：大牌判定与潇洒滑动窗口计算
# ==========================================
def is_qing_yi_se(hand, melds):
    all_tiles = hand + [m[1] for m in melds]
    if any(t >= ZHONG for t in all_tiles): return False
    suits = set(t // 9 for t in all_tiles)
    return len(suits) == 1

def is_hun_yi_se(hand, melds):
    all_tiles = hand + [m[1] for m in melds]
    if not any(t >= ZHONG for t in all_tiles) or not any(t < ZHONG for t in all_tiles): return False
    suits = set(t // 9 for t in all_tiles if t < ZHONG)
    return len(suits) == 1

def is_yi_tiao_long(hand):
    # 修复：必须能剥离出 1-9，且剩下的 5 张牌能组成 1面子 + 1将牌
    for suit in range(3):
        start = suit * 9
        if all((start + i) in hand for i in range(9)):
            h_copy = hand[:]
            for i in range(9): h_copy.remove(start + i)
            if len(h_copy) == 5 and check_standard_hu(h_copy): return True
    return False

def is_da_diao_che(hand, melds):
    return len(melds) == 4 and len(hand) == 2

def is_peng_peng_hu(hand, melds):
    if len(hand) % 3 != 2: return False
    for t, c in Counter(hand).items():
        if c not in (2, 3): return False
    return True

def check_seven_fairies(hand):
    if not is_seven_pairs(hand): return False
    if not is_qing_yi_se(hand, []): return False
    t_set = sorted(list(set(hand)))
    if len(t_set) != 7: return False
    return t_set[-1] % 9 - t_set[0] % 9 == 6

def check_hong_qi_piao_piao(hand, melds, win_tile, is_duidao_hu):
    c = Counter(hand)
    m_tiles = [m[1] for m in melds]
    if ZHONG in m_tiles and BAI in m_tiles: return True
    if c[ZHONG] >= 3 and c[BAI] >= 3: return True
    if is_duidao_hu and ((c[ZHONG] >= 3 and c[BAI] >= 2 and win_tile == BAI) or (c[BAI] >= 3 and c[ZHONG] >= 2 and win_tile == ZHONG)): return True
    if is_seven_pairs(hand) and c[ZHONG] >= 2 and c[BAI] >= 2: return True
    return False

def count_xiaosa_pairs(hand):
    # 潇洒核心修复：滑动窗口统计 3 连对
    counts = Counter(hand)
    pairs = [t for t, c in counts.items() if c >= 2]
    xiaosa_count = 0
    for suit in range(3):
        suit_pairs = set(t for t in pairs if t // 9 == suit and t < ZHONG)
        for i in range(9 - 2):
            t = suit * 9 + i
            if t in suit_pairs and (t+1) in suit_pairs and (t+2) in suit_pairs:
                xiaosa_count += 1
    return xiaosa_count

# ==========================================
# 🛑 核心修复 3：算番逻辑严格剥离 (平胡与大胡互斥)
# ==========================================
def calculate_final_score(player, win_tile, is_zimo, context_flags):
    hand = sorted(player["hand"])
    melds = player["melds"]
    fa_count = player["fa_count"]
    
    fan = 0
    big_patterns = []
    
    is_sp = is_seven_pairs(hand)
    is_pp = is_peng_peng_hu(hand, melds)
    is_sf = check_seven_fairies(hand) if is_sp else False
    
    # 1. 判定所有大胡
    if is_sf: big_patterns.append(("七仙女", 50))
    elif is_sp: big_patterns.append(("七对", 10))
    elif is_pp: big_patterns.append(("碰碰胡", 10))
        
    if is_qing_yi_se(hand, melds): big_patterns.append(("清一色", 15))
    elif is_hun_yi_se(hand, melds): big_patterns.append(("混一色", 10))
    if is_yi_tiao_long(hand): big_patterns.append(("一条龙", 10))
    if is_da_diao_che(hand, melds): big_patterns.append(("大吊车", 15))
    if check_hong_qi_piao_piao(hand, melds, win_tile, context_flags.get("is_duidao_hu", False)): 
        big_patterns.append(("红旗飘飘", 10))
        
    if context_flags.get("haidi"): big_patterns.append(("海底", 10))
    if context_flags.get("gang_qiang"): big_patterns.append(("杠上开花/抢杠", 10))
    if context_flags.get("tianhu"): big_patterns.append(("天胡", 15))
    if context_flags.get("dihu"): big_patterns.append(("地胡", 20))
    
    # 潇洒与豪华 (独立附加分)
    if is_sp and not is_sf:
        xc = count_xiaosa_pairs(hand)
        if xc > 0: big_patterns.append((f"潇洒x{xc}", xc * 5))
        hc = sum(1 for c in Counter(hand).values() if c == 4)
        if hc > 0: big_patterns.append((f"豪华x{hc}", hc * 5))

    is_da_hu = len(big_patterns) > 0
    da_hu_fan = sum(p[1] for p in big_patterns)
    fan += da_hu_fan
    
    # 2. 基础番数：大胡不再累加平胡的底番 1 番
    if not is_da_hu:
        fan += 1
        
    # 3. 门前清 (没有任何明牌碰杠)
    is_men_qian_qing = len(melds) == 0
    if is_men_qian_qing:
        fan += 1
        
    # 4. 自摸加成 (平胡与大胡严格区分)
    if is_zimo:
        if is_sf: fan += 50; big_patterns.append(("自摸(仙女)", 50))
        elif is_da_hu: fan += 5; big_patterns.append(("自摸(大胡)", 5))
        else: fan += 1; big_patterns.append(("自摸", 1))
            
    # 5. 发财番
    if fa_count >= 4: fan += 10; big_patterns.append(("四发财", 10))
    else: fan += fa_count
        
    # 6. 字牌暗刻/明杠番
    hand_c = Counter(hand)
    for m_type, m_tile in melds:
        if m_tile in (ZHONG, BAI):
            if m_type == "PONG": fan += 1
            elif m_type == "KONG": fan += 3
        else:
            if m_type == "KONG": fan += 1 
            
    if hand_c[ZHONG] >= 3: fan += 1
    if hand_c[BAI] >= 3: fan += 1
    
    return fan, big_patterns

def simulate_game(game_id, record_dataset=False):
    """单局游戏核心流程模拟"""
    deck = create_deck()
    players = [{"hand": [], "melds": [], "fa_count": 0, "is_tenpai": False} for _ in range(4)]
    global_discard_pool = []
    discards_by_player = [[], [], [], []]
    last_tile = None
    
    for _ in range(13):
        for p in players:
            p["hand"].append(deck.pop())
            
    for p in players:
        while FA in p["hand"]:
            p["hand"].remove(FA)
            p["fa_count"] += 1
            if deck:
                p["hand"].append(deck.pop(0))
                
    first_tenpai_turn = -1
    win_turn = -1
    win_player = -1
    
    turns = 0
    current_player = 0
    
    local_S_list = []
    local_A_list = []
    
    while deck:
        turns += 1
        p = players[current_player]
        
        drawn_tile = deck.pop()
        
        while drawn_tile == FA and deck:
            p["fa_count"] += 1
            drawn_tile = deck.pop(0)
            
        p["hand"].append(drawn_tile)
        last_tile = drawn_tile
        
        s_num = calculate_shanten_accurate(p["hand"], p["melds"])
        if s_num == -1:
            total_p, _ = calculate_final_score(p, drawn_tile, True, {})
            if total_p >= 2:
                win_turn = turns
                win_player = current_player
                break
                
        if s_num == 0 and not p["is_tenpai"]:
            p["is_tenpai"] = True
            if first_tenpai_turn == -1:
                first_tenpai_turn = turns
                
        if record_dataset:
            tot_fa = sum(pl["fa_count"] for pl in players)
            opp_melds = []
            for idx in range(4):
                if idx != current_player: opp_melds.extend(players[idx]["melds"])
                
            rel_discards = [
                discards_by_player[current_player],
                discards_by_player[(current_player + 1) % 4],
                discards_by_player[(current_player + 2) % 4],
                discards_by_player[(current_player + 3) % 4]
            ]
                
            S_tensor = encode_state(p["hand"], p["melds"], rel_discards, opp_melds, tot_fa, len(deck), last_tile)
            local_S_list.append(S_tensor)
        
        discarded_tile = get_best_discard_smart(p["hand"], p["melds"])
        
        if record_dataset:
            local_A_list.append(discarded_tile)
            
        p["hand"].remove(discarded_tile)
        
        hu_triggered = False
        for i in range(1, 4):
            other_idx = (current_player + i) % 4
            op = players[other_idx]
            op["hand"].append(discarded_tile)
            if calculate_shanten_accurate(op["hand"], op["melds"]) == -1:
                total_p_other, _ = calculate_final_score(op, discarded_tile, False, {})
                if total_p_other >= 2:
                    win_turn = turns
                    win_player = other_idx
                    hu_triggered = True
                    op["hand"].remove(discarded_tile)
                    break
            op["hand"].remove(discarded_tile)
            
        if hu_triggered:
            break
        
        action_taken = False
        for i in range(1, 4):
            other_p_idx = (current_player + i) % 4
            other_p = players[other_p_idx]
            count_in_hand = other_p["hand"].count(discarded_tile)
            
            if count_in_hand >= 3: 
                last_tile = discarded_tile 
                other_p["hand"] = [t for t in other_p["hand"] if t != discarded_tile]
                other_p["melds"].append(("KONG", discarded_tile))
                if deck: 
                    bonus_tile = deck.pop(0)
                    while bonus_tile == FA:
                        other_p["fa_count"] += 1
                        if not deck:
                            bonus_tile = None
                            break
                        bonus_tile = deck.pop(0)
                    if bonus_tile is not None:
                        other_p["hand"].append(bonus_tile)
                        last_tile = bonus_tile
                
                if other_p["hand"]:
                    new_discard = get_best_discard_smart(other_p["hand"], other_p["melds"])
                    other_p["hand"].remove(new_discard)
                    
                current_player = other_p_idx
                action_taken = True
                break
                
            elif count_in_hand == 2 and should_pong(other_p["hand"], discarded_tile): 
                last_tile = discarded_tile
                other_p["hand"] = [t for t in other_p["hand"] if t != discarded_tile]
                other_p["melds"].append(("PONG", discarded_tile))
                
                new_discard = get_best_discard_smart(other_p["hand"], other_p["melds"])
                other_p["hand"].remove(new_discard)
                
                current_player = other_p_idx 
                action_taken = True
                break
        
        if not action_taken:
            global_discard_pool.append(discarded_tile)
            discards_by_player[current_player].append(discarded_tile)
            current_player = (current_player + 1) % 4
            
    res_map = {
        "game_id": game_id,
        "total_turns": turns,
        "tenpai_turn": first_tenpai_turn,
        "win_turn": win_turn,
        "is_drawn": win_turn == -1
    }
    
    if record_dataset:
        return res_map, local_S_list, local_A_list
    return res_map

def run_simulation(num_games):
    start_time = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] 启动蒙特卡洛模拟...")
    print(f"目标局数: {num_games} 局")
    
    cpu_cores = multiprocessing.cpu_count()
    print(f"已分配并行进程池核心数: {cpu_cores} 核")
    
    # 因为全局字典会被所有子进程继承引用（fork模式下，Windows 是 spawn 但字典由于在模块顶部可被加载），
    # 在拥有超小型字典后，速度应该会直接超神
    with multiprocessing.Pool(processes=cpu_cores) as pool:
        results = pool.map(simulate_game, range(num_games))
        
    end_time = time.time()
    avg_turns = sum(r["total_turns"] for r in results) / num_games
    drawn_games = sum(1 for r in results if r["is_drawn"])
    win_games = num_games - drawn_games
    win_rate = (win_games / num_games) * 100
    
    tenpai_games = [r["tenpai_turn"] for r in results if r["tenpai_turn"] != -1]
    avg_tenpai_turn = sum(tenpai_games) / len(tenpai_games) if tenpai_games else -1
    
    win_turns = [r["win_turn"] for r in results if r["win_turn"] != -1]
    avg_win_turn = sum(win_turns) / len(win_turns) if win_turns else -1
    
    print("\n" + "="*40)
    print("🎯 测试数据汇总")
    print("="*40)
    print(f"总耗时: {end_time - start_time:.2f} 秒")
    print(f"每秒运算极速: {num_games / (end_time - start_time):.0f} 局/秒")
    print(f"单局平均摸打总次数 (Turns): {avg_turns:.1f} 次")
    print(f"流局率: {(drawn_games / num_games) * 100:.1f}%")
    print(f"胡牌率: {win_rate:.1f}%")
    if avg_tenpai_turn != -1:
        print(f"平均听牌回合: {avg_tenpai_turn:.1f}")
    if avg_win_turn != -1:
        print(f"平均胡牌回合: {avg_win_turn:.1f}")
    print("="*40)

def simulate_game_wrapper(game_id):
    return simulate_game(game_id, record_dataset=True)

def generate_sl_dataset(num_games=1000):
    start_time = time.time()
    out_file = "ruichang_expert_v1.npz"
    cpu_cores = multiprocessing.cpu_count()
    print(f"\n[{time.strftime('%H:%M:%S')}] 启动【上帝视角数据集】提取流水线 ({cpu_cores}核并发)...")
    print(f"目标挖掘局数: {num_games} 局")
    
    all_S = []
    all_A = []
    
    with multiprocessing.Pool(processes=cpu_cores) as pool:
        results = pool.map(simulate_game_wrapper, range(num_games))
        
    for r in results:
        res_info, s_list, a_list = r
        all_S.extend(s_list)
        all_A.extend(a_list)
        
    if not all_S:
        print("未提取到任何数据。")
        return
        
    S_arr = np.array(all_S, dtype=np.float32)
    A_arr = np.array(all_A, dtype=np.int32)
    
    end_time = time.time()
    print(f"提取完成！总耗时: {end_time-start_time:.2f} 秒")
    print(f"共生成极高质量动作序列 (S, A) 样本: {len(S_arr)} 条！")
    print(f"特征矩阵 S Shape: {S_arr.shape}")
    print(f"动作标签 A Shape: {A_arr.shape}")
        
    np.savez_compressed(out_file, S=S_arr, A=A_arr)
    print(f"数据集已无损压缩保存至 -> {out_file} ({os.path.getsize(out_file) / 1024 / 1024:.2f} MB)\n")

if __name__ == "__main__":
    generate_sl_dataset(1000)