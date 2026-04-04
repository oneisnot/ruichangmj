from collections import Counter

ZHONG = 27
BAI = 28
FA = 29

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
    # 必须在手牌里完整的一二三四五六七八九（由于不能吃牌）
    for suit in range(3):
        start = suit * 9
        if all((start + i) in hand for i in range(9)):
            # 需要验证剩下的 5 张牌能否和这 9 张牌有效组成胡牌结构？
            # 实际上只要胡牌且包含1~9，就算一条龙
            return True
    return False

def is_da_diao_che(hand, melds):
    return len(melds) == 4 and len(hand) == 2  # hand contains win_tile and 1 in-hand tile

def is_seven_pairs(hand):
    if len(hand) != 14:
        return False
    c = Counter(hand)
    return all(count in (2, 4) for count in c.values())

def is_peng_peng_hu(hand, melds):
    # 只有刻子和将牌
    if len(hand) % 3 != 2: return False
    c = Counter(hand)
    pairs = 0
    tris = 0
    for count in c.values():
        if count == 2:
            pairs += 1
        elif count == 3:
            tris += 1
        elif count == 4: # 当作一个碰和一个单张是不行的
            return False
        else:
            return False
    return pairs == 1

def check_seven_fairies(hand):
    """验证七仙女：极其苛刻的清一色顺连七对"""
    if not is_seven_pairs(hand):
        return False
    if not is_qing_yi_se(hand, []):
        return False
    t_set = sorted(list(set(hand)))
    if len(t_set) != 7: return False
    
    # 必须是一门连续的7个数字
    suit = t_set[0] // 9
    for t in t_set:
        if t // 9 != suit: return False
        
    start_val = t_set[0] % 9
    end_val = t_set[-1] % 9
    if end_val - start_val == 6 and len(t_set) == 7:
        return True
    return False

def check_hong_qi_piao_piao(hand, melds, win_tile, is_duidao_hu):
    """
    红旗飘飘
    1. 碰杠区里有中和白
    2. 手里有中暗刻、白暗刻
    3. 中白其中一种成刻，另一种是一对，对倒胡牌时胡了这一对
    4. 在做七对时，包含一对中一对白
    """
    c = Counter(hand)
    m_tiles = [m[1] for m in melds]
    
    # 条件1：全部碰/杠出
    if ZHONG in m_tiles and BAI in m_tiles:
        return True
        
    # 条件2：双暗刻
    if c[ZHONG] >= 3 and c[BAI] >= 3:
        return True
        
    # 条件3：对倒胡且互换
    if is_duidao_hu:
        if (c[ZHONG] >= 3 and c[BAI] >= 2 and win_tile == BAI) or \
           (c[BAI] >= 3 and c[ZHONG] >= 2 and win_tile == ZHONG):
            return True
            
    # 条件4：七对带中白
    if is_seven_pairs(hand):
        if c[ZHONG] >= 2 and c[BAI] >= 2:
            return True
            
    return False

def count_xiaosa(hand, melds, is_seven_pairs_flag, is_peng_peng_hu_flag):
    """计算潇洒（滚）次数"""
    gun_count = 0
    if is_seven_pairs_flag:
        t_set = sorted([t for t, count in Counter(hand).items() if count >= 2])
        # 找顺连对子，比如 11 22, 实际上只要 t_set 中有连续相邻牌且同花色即可
        i = 0
        while i < len(t_set) - 1:
            if t_set[i] < ZHONG and t_set[i+1] == t_set[i] + 1 and t_set[i]//9 == t_set[i+1]//9:
                gun_count += 1
            i += 1
            
    if is_peng_peng_hu_flag:
        # 只计算明/暗杠！碰牌不算。
        gangs = sorted([m[1] for m in melds if m[0] == "KONG"])
        i = 0
        while i < len(gangs) - 1:
            if gangs[i] < ZHONG and gangs[i+1] == gangs[i] + 1 and gangs[i]//9 == gangs[i+1]//9:
                gun_count += 1
            i += 1
            
    return gun_count

def count_haohua(hand):
    """计算豪华（4张充当2个对子）"""
    return sum(1 for count in Counter(hand).values() if count == 4)

def calculate_final_score(player, win_tile, is_zimo, context_flags):
    hand = sorted(player["hand"])
    melds = player["melds"]
    fa_count = player["fa_count"]
    
    pts = 1  # 胡牌底分
    pts += context_flags.get("haidi", 0) * 10
    pts += context_flags.get("gang_qiang", 0) * 10
    
    # --- 1. 基础分核算 ---
    if fa_count >= 4:
        pts += 10
    else:
        pts += fa_count
        
    if not melds:
        pts += 1
        
    pts += 1 if context_flags.get("ting_dan_zhang", False) else 0
    if is_zimo: pts += 1
    
    hand_c = Counter(hand)
    for m_type, m_tile in melds:
        if m_tile in (ZHONG, BAI):
            if m_type == "PONG": pts += 1
            elif m_type == "KONG": pts += 3 # 明杠3，暗杠4(此处简略暗杠判定)
        else:
            if m_type == "KONG": pts += 1

    if hand_c[ZHONG] >= 3: pts += 1
    if hand_c[BAI] >= 3: pts += 1

    # --- 2. 大牌特征触发 ---
    big_patterns = []
    
    is_sp = is_seven_pairs(hand)
    is_pp = is_peng_peng_hu(hand, melds)
    is_sf = check_seven_fairies(hand) if is_sp else False
    
    # 清/混一色
    if is_qing_yi_se(hand, melds): big_patterns.append(("清一色", 15))
    elif is_hun_yi_se(hand, melds): big_patterns.append(("混一色", 10))
    
    if is_yi_tiao_long(hand): big_patterns.append(("一条龙", 10))
    if is_da_diao_che(hand, melds): big_patterns.append(("大吊车", 15))
    if check_hong_qi_piao_piao(hand, melds, win_tile, context_flags.get("is_duidao_hu", False)): 
        big_patterns.append(("红旗飘飘", 10))
        
    if is_sf:
        big_patterns.append(("七仙女", 50))
        if is_zimo: pts += 50 # 额外自摸50炮
    elif is_sp:
        big_patterns.append(("七对", 10))
    elif is_pp:
        big_patterns.append(("碰碰胡", 10))
        
    if context_flags.get("tianhu"): big_patterns.append(("天胡", 20))
    if context_flags.get("dihu"): big_patterns.append(("地胡", 20))
    
    gun_count = count_xiaosa(hand, melds, is_sp and not is_sf, is_pp)
    haohua_count = count_haohua(hand) if is_sp else 0
    
    if gun_count > 0: big_patterns.append((f"潇洒x{gun_count}", gun_count * 5))
    if haohua_count > 0: big_patterns.append((f"豪华x{haohua_count}", haohua_count * 5))

    has_hua = len(big_patterns) > 0
    for name, p in big_patterns:
        pts += p
        
    if is_zimo and has_hua:
        pts += 5 # 大牌自摸暴击

    return pts, big_patterns

def test():
    # 测试例 1: 完美七仙女 (11 22 33 44 55 66 77 万) 自摸
    hand = [0,0, 1,1, 2,2, 3,3, 4,4, 5,5, 6,6]
    player = {"hand": hand, "melds": [], "fa_count": 0}
    pts, p = calculate_final_score(player, 6, True, {})
    print(f"测试1 - 七仙女自摸: 炮数: {pts}, 牌型: {p}")
    
    # 测试例 2: 豪华红旗飘飘七对 (11 22 33 44 44 27 27 28 28)
    hand = [1,1, 2,2, 3,3, 4,4, 4,4, 27,27, 28,28]
    player = {"hand": hand, "melds": [], "fa_count": 2}
    pts, p = calculate_final_score(player, 28, False, {})
    print(f"测试2 - 豪华红旗飘飘: 炮数: {pts}, 牌型: {p}")
    
if __name__ == "__main__":
    test()
