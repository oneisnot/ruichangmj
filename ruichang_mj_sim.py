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
    print(f"[{time.strftime('%H:%M:%S')}] 加载向听数离线查表 (DP Lookup)...")
    with open(CACHE_PATH, "rb") as f:
        cache_data = pickle.load(f)
        SUIT_TABLE = cache_data["suit_table"]
        HONOR_TABLE = cache_data["honor_table"]
else:
    print("警告: 找不到 ruichang_mj_shanten_cache.pkl，请先运行 generate_shanten_table.py！")

def create_deck():
    """初始化 120 张牌：0-8万，9-17条，18-26饼，27红中，28白板，29发财"""
    deck = []
    for i in range(30):
        deck.extend([i] * 4)
    random.shuffle(deck)
    return deck

def evaluate_tile_value(tile, hand):
    """
    牌效估值器：分值越低，说明这张牌越没用，越应该被优先打掉。
    """
    if tile == FA:
        return 1000 # 发财绝对不打
        
    count = hand.count(tile)
    if count >= 2:
        return 100  # 有对子或暗刻，保留优先级极高
    
    # 评估顺子搭子的潜力 (字牌没有顺子)
    neighbors = 0
    if tile < ZHONG:
        if tile % 9 > 0 and (tile - 1) in hand: neighbors += 1
        if tile % 9 < 8 and (tile + 1) in hand: neighbors += 1
        if tile % 9 > 1 and (tile - 2) in hand: neighbors += 1
        if tile % 9 < 7 and (tile + 2) in hand: neighbors += 1
        
    # 如果是红中白板的单张，给予一定的保留权重 (博红旗飘飘或起胡底分)
    if tile in (ZHONG, BAI):
        return 15
        
    return neighbors * 10

def get_best_discard(hand):
    """根据牌效估值，找出最差的一张牌打掉"""
    worst_tile = hand[0]
    lowest_val = float('inf')
    
    # 避免重复计算
    unique_tiles = set(hand)
    for tile in unique_tiles:
        val = evaluate_tile_value(tile, hand)
        if val < lowest_val:
            lowest_val = val
            worst_tile = tile
            
    return worst_tile

def should_pong(hand, discarded_tile):
    """
    判断是否应该碰牌的启发式老手逻辑（核心防守与做牌策略）
    """
    # 统计手牌中的对子数量
    counts = Counter(hand)
    pairs = [t for t, c in counts.items() if c >= 2]
    num_pairs = len(pairs)
    
    # 【逻辑门 1】：做七对的潜力保护
    if num_pairs >= 4:
        return False
        
    # 【逻辑门 2】：唯一将牌保护
    if num_pairs == 1 and discarded_tile in pairs:
        return False
        
    # 【逻辑门 3】：顺子搭子防破坏
    has_sequence_neighbor = False
    if discarded_tile < ZHONG:
        if discarded_tile % 9 > 0 and (discarded_tile - 1) in hand:
            has_sequence_neighbor = True
        if discarded_tile % 9 < 8 and (discarded_tile + 1) in hand:
            has_sequence_neighbor = True
            
    if has_sequence_neighbor:
        return False
        
    # 【逻辑门 4】：字牌特权与常规碰牌
    return True

# --- 新增向听数和算分逻辑 ---

def calculate_shanten_accurate(hand, melds):
    """
    O(1) 查表法的真实向听数算法。
    """
    meld_count = len(melds)
    shanten_7 = 100
    if meld_count == 0:
        counts = Counter(hand)
        num_pairs = sum(1 for c in counts.values() if c >= 2)
        num_kinds = len(counts)
        shanten_7 = 6 - num_pairs + max(0, 7 - num_kinds)

    counts = [0] * 30
    for t in hand: 
        counts[t] += 1
        
    w_sum = t_sum = b_sum = h_sum = 0
    w_p = t_p = b_p = h_p = 1
    for i in range(9):
        w_sum += counts[i] * w_p
        w_p *= 5
        t_sum += counts[9+i] * t_p
        t_p *= 5
        b_sum += counts[18+i] * b_p
        b_p *= 5
    for i in range(3):
        h_sum += counts[27+i] * h_p
        h_p *= 5

    # 查询哈希表
    w_res = SUIT_TABLE.get(w_sum, ((-1, -1, -1, -1, -1), (-1, -1, -1, -1, -1)))
    t_res = SUIT_TABLE.get(t_sum, ((-1, -1, -1, -1, -1), (-1, -1, -1, -1, -1)))
    b_res = SUIT_TABLE.get(b_sum, ((-1, -1, -1, -1, -1), (-1, -1, -1, -1, -1)))
    h_res = HONOR_TABLE.get(h_sum, ((-1, -1, -1, -1, -1), (-1, -1, -1, -1, -1)))
    
    def merge(res1, res2):
        out_no = [-1] * 5
        out_pair = [-1] * 5
        rn1, rp1 = res1
        rn2, rp2 = res2
        
        for g1 in range(5):
            t1_no = rn1[g1]
            t1_p = rp1[g1]
            if t1_no == -1 and t1_p == -1: continue
            
            for g2 in range(5 - g1):
                t2_no = rn2[g2]
                t2_p = rp2[g2]
                if t2_no == -1 and t2_p == -1: continue
                
                if t1_no != -1 and t2_no != -1:
                    val = t1_no + t2_no
                    if val > out_no[g1+g2]: out_no[g1+g2] = val
                    
                if t1_p != -1 and t2_no != -1:
                    val = t1_p + t2_no
                    if val > out_pair[g1+g2]: out_pair[g1+g2] = val
                    
                if t1_no != -1 and t2_p != -1:
                    val = t1_no + t2_p
                    if val > out_pair[g1+g2]: out_pair[g1+g2] = val
                    
        for g in range(5):
            if out_no[g] > 4 - g: out_no[g] = 4 - g
            if out_pair[g] > 4 - g: out_pair[g] = 4 - g
            
        return (out_no, out_pair)
        
    m1 = merge(w_res, t_res)
    m2 = merge(m1, b_res)
    m3 = merge(m2, h_res)
    
    min_s = 8
    fn, fp = m3
    for g in range(5):
        if fn[g] != -1:
            s_val = 8 - 2*(g + meld_count) - fn[g]
            if s_val < min_s: min_s = s_val
        if fp[g] != -1:
            s_val = 8 - 2*(g + meld_count) - fp[g] - 1
            if s_val < min_s: min_s = s_val
            
    return min(shanten_7, min_s)

def get_best_discard_smart(hand, melds):
    """基于向听数驱动的真实牌效弃牌"""
    unique_tiles = list(set(hand))
    best_discard = unique_tiles[0]
    min_s = 100
    
    lowest_base_val = float('inf') 
    
    for tile in unique_tiles:
        hand.remove(tile)
        s_num = calculate_shanten_accurate(hand, melds) 
        hand.append(tile)
        
        if s_num < min_s:
            min_s = s_num
            best_discard = tile
            lowest_base_val = evaluate_tile_value(tile, hand)
        elif s_num == min_s:
            base_val = evaluate_tile_value(tile, hand)
            if base_val < lowest_base_val:
                lowest_base_val = base_val
                best_discard = tile
                
    return best_discard

def is_qing_yi_se(hand, melds):
    all_tiles = hand + [m[1] for m in melds]
    if any(t >= ZHONG for t in all_tiles):
        return False
    suits = set(t // 9 for t in all_tiles)
    return len(suits) == 1

def is_hun_yi_se(hand, melds):
    all_tiles = hand + [m[1] for m in melds]
    has_honors = any(t >= ZHONG for t in all_tiles)
    has_suits = any(t < ZHONG for t in all_tiles)
    if not has_honors or not has_suits:
        return False
    suits = set(t // 9 for t in all_tiles if t < ZHONG)
    return len(suits) == 1

def is_yi_tiao_long(hand):
    for suit in range(3):
        start = suit * 9
        if all((start + i) in hand for i in range(9)):
            return True
    return False

def is_da_diao_che(hand, melds):
    return len(melds) == 4 and len(hand) == 2

def is_seven_pairs(hand):
    if len(hand) != 14: return False
    c = Counter(hand)
    return all(count in (2, 4) for count in c.values())

def is_peng_peng_hu(hand, melds):
    if len(hand) % 3 != 2: return False
    c = Counter(hand)
    pairs = tris = 0
    for count in c.values():
        if count == 2: pairs += 1
        elif count == 3: tris += 1
        else: return False
    return pairs == 1

def check_seven_fairies(hand):
    if not is_seven_pairs(hand): return False
    if not is_qing_yi_se(hand, []): return False
    t_set = sorted(list(set(hand)))
    if len(t_set) != 7: return False
    suit = t_set[0] // 9
    for t in t_set:
        if t // 9 != suit: return False
    if t_set[-1] % 9 - t_set[0] % 9 == 6 and len(t_set) == 7:
        return True
    return False

def check_hong_qi_piao_piao(hand, melds, win_tile, is_duidao_hu):
    c = Counter(hand)
    m_tiles = [m[1] for m in melds]
    if ZHONG in m_tiles and BAI in m_tiles: return True
    if c[ZHONG] >= 3 and c[BAI] >= 3: return True
    if is_duidao_hu:
        if (c[ZHONG] >= 3 and c[BAI] >= 2 and win_tile == BAI) or \
           (c[BAI] >= 3 and c[ZHONG] >= 2 and win_tile == ZHONG):
            return True
    if is_seven_pairs(hand) and c[ZHONG] >= 2 and c[BAI] >= 2: return True
    return False

def count_xiaosa(hand, melds, is_sp_flag, is_pp_flag):
    gun_count = 0
    if is_sp_flag:
        t_set = sorted([t for t, count in Counter(hand).items() if count >= 2])
        i = 0
        while i < len(t_set) - 1:
            if t_set[i] < ZHONG and t_set[i+1] == t_set[i] + 1 and t_set[i]//9 == t_set[i+1]//9:
                gun_count += 1
            i += 1
    if is_pp_flag:
        # 刻就算了。遍历已有明刻/杠和手牌暗刻
        triplets = [m[1] for m in melds if m[0] in ("PONG", "KONG")]
        triplets.extend([t for t, count in Counter(hand).items() if count >= 3])
        triplets = sorted(list(set(triplets)))
        
        i = 0
        while i < len(triplets) - 1:
            if triplets[i] < ZHONG and triplets[i+1] == triplets[i] + 1 and triplets[i]//9 == triplets[i+1]//9:
                gun_count += 1
            i += 1
    return gun_count

def calculate_final_score(player, win_tile, is_zimo, context_flags):
    """
    最全番数/炮数精准计算结算系统
    """
    hand = sorted(player["hand"])
    melds = player["melds"]
    fa_count = player["fa_count"]
    
    pts = 1  
    pts += context_flags.get("haidi", 0) * 10
    pts += context_flags.get("gang_qiang", 0) * 10
    
    if fa_count >= 4: pts += 10
    else: pts += fa_count
        
    if not melds: pts += 1
    if context_flags.get("ting_dan_zhang", False): pts += 1
    if is_zimo: pts += 1
    
    hand_c = Counter(hand)
    for m_type, m_tile in melds:
        if m_tile in (ZHONG, BAI):
            if m_type == "PONG": pts += 1
            elif m_type == "KONG": pts += 3 
        else:
            if m_type == "KONG": pts += 1

    if hand_c[ZHONG] >= 3: pts += 1
    if hand_c[BAI] >= 3: pts += 1

    big_patterns = []
    is_sp = is_seven_pairs(hand)
    is_pp = is_peng_peng_hu(hand, melds)
    is_sf = check_seven_fairies(hand) if is_sp else False
    
    if is_qing_yi_se(hand, melds): big_patterns.append(("清一色", 15))
    elif is_hun_yi_se(hand, melds): big_patterns.append(("混一色", 10))
    
    if is_yi_tiao_long(hand): big_patterns.append(("一条龙", 10))
    if is_da_diao_che(hand, melds): big_patterns.append(("大吊车", 15))
    if check_hong_qi_piao_piao(hand, melds, win_tile, context_flags.get("is_duidao_hu", False)): 
        big_patterns.append(("红旗飘飘", 10))
        
    zimo_extra = 0
    if is_sf:
        big_patterns.append(("七仙女", 50))
        if is_zimo: zimo_extra = max(zimo_extra, 50) 
    elif is_sp:
        big_patterns.append(("七对", 10))
    elif is_pp:
        big_patterns.append(("碰碰胡", 10))
        
    if context_flags.get("tianhu"): big_patterns.append(("天胡", 15))
    if context_flags.get("dihu"): big_patterns.append(("地胡", 20))
    
    gun_count = count_xiaosa(hand, melds, is_sp and not is_sf, is_pp)
    haohua_count = sum(1 for count in hand_c.values() if count == 4) if is_sp else 0
    
    if gun_count > 0: big_patterns.append((f"潇洒x{gun_count}", gun_count * 5))
    if haohua_count > 0: big_patterns.append((f"豪华x{haohua_count}", haohua_count * 5))

    has_hua = len(big_patterns) > 0
    for name, p in big_patterns:
        pts += p
        
    if is_zimo and has_hua:
        zimo_extra = max(zimo_extra, 5)
        
    pts += zimo_extra

    return pts, big_patterns

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