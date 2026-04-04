import numpy as np
from collections import Counter

def encode_state(player_hand, player_melds, discards_by_player, opponents_melds, fa_revealed_total, deck_len, last_tile, dealer_idx=0, base_score=5):
    """
    将全局棋盘状态编码为 3D Numpy Array 张量 (16 x 30)。
    
    通道分配:
    Channel 0-13: [同原版保持一致] 包含手牌、鸣牌、牌池、掩码等
    Channel 14: 庄家位置信号盘 (Dealer Position)
    Channel 15: 本局底分规则信号 (Conditioning: Base Score / 20.0)
    """
    state_matrix = np.zeros((16, 30), dtype=np.float32)
    
    # 1. 玩家手牌分布编码 (Thermometer Encoding)
    hand_counts = Counter(player_hand)
    for tile, count in hand_counts.items():
        if tile >= 30: continue
        if count >= 1: state_matrix[0][tile] = 1.0
        if count >= 2: state_matrix[1][tile] = 1.0
        if count >= 3: state_matrix[2][tile] = 1.0
        if count >= 4: state_matrix[3][tile] = 1.0
        
        # 通道13：合法弃牌掩码 
        state_matrix[13][tile] = 1.0 
        
    # 2. 玩家自己的鸣牌 (碰/杠)
    for m_type, m_tile in player_melds:
        if m_tile < 30:
            state_matrix[4][m_tile] = 1.0
            
    # 3. 极其精细的四家废牌池 (0:己方, 1:下家, 2:对家, 3:上家)
    for rel_idx, pool in enumerate(discards_by_player):
        d_counts = Counter(pool)
        for tile, count in d_counts.items():
            if tile < 30:
                state_matrix[5 + rel_idx][tile] = count / 4.0 
            
    # 4. 对手鸣牌
    for m_type, m_tile in opponents_melds:
        if m_tile < 30:
            state_matrix[9][m_tile] = 1.0
            
    # 5. 最后一张牌（触发器）
    if last_tile is not None and last_tile < 30:
        state_matrix[10][last_tile] = 1.0
            
    # 6. 全局常量背景 
    state_matrix[11][:] = float(fa_revealed_total) / 4.0 
    state_matrix[12][:] = float(deck_len) / 120.0
    
    # 7. 【新增阶段】庄家位置：P0庄(1.0), 下家庄(0.25), 对家(0.5), 上家(0.75)
    # 计算当前庄家相对于 Player 0 的偏移
    if dealer_idx == 0: val = 1.0
    elif dealer_idx == 1: val = 0.25
    elif dealer_idx == 2: val = 0.50
    else: val = 0.75 # dealer_idx == 3
    state_matrix[14][:] = val
    
    # 8. 【新增阶段】底分规则特征 (0-20分归一化)
    state_matrix[15][:] = float(base_score) / 20.0
    
    return state_matrix

def print_tensor_summary(state_matrix):
    print(f"✅ State Encoder Output 成功, (Shape: {state_matrix.shape})")
    print("-" * 50)
    channels = ["Hand >= 1", "Hand >= 2", "Hand >= 3", "Hand == 4", 
                "My Melds", "My Discards", "Next Discards", "Across Discards", "Prev Discards",
                "Opponent Melds", "Last target Tile", "Fa Status", "Deck Countdown", "Legal Masks"]
    for i in range(14):
        active_coords = np.where(state_matrix[i] > 0)[0]
        if len(active_coords) > 0:
            vals = [f"T{t}:{state_matrix[i][t]:.2f}" for t in active_coords[:8]]
            plus = " ..." if len(active_coords) > 8 else ""
            print(f"[{i:2d}] {channels[i]:<18} | " + ", ".join(vals) + plus)
    print("-" * 50)
            
if __name__ == "__main__":
    print("构建测试张量转化...")
    test_hand = [0, 0, 1, 1, 1, 10, 27, 28, 28, 28, 28] 
    test_melds = [("PONG", 4)] 
    
    # 模拟四个人的牌池 (必须传入相对位置，分别是 [自己, 下家, 对家, 上家])
    test_discards = [
        [15, 16],        # 自己
        [8, 8, 8, 29],   # 下家疯狂打8万
        [27],            # 对家
        [0, 1, 2]        # 上家
    ]
    
    test_opp_melds = [("KONG", 22), ("PONG", 10)] 
    fa_tot = 3 
    deck_remaining = 35
    last_action_tile = 8 # 假设这张牌是刚刚产生的
    
    mat = encode_state(test_hand, test_melds, test_discards, test_opp_melds, fa_tot, deck_remaining, last_action_tile)
    print_tensor_summary(mat)
