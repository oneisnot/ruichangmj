import random
import numpy as np
from collections import Counter
import ruichang_mj_sim as sim
from state_encoder import encode_state

class RuichangMahjongEnv:
    """
    瑞昌麻将 PPO 训练环境封装
    
    固定视角: 训练进程始终扮演 Player 0。
    其余 3 名对手 (Player 1, 2, 3) 使用内置的启发式专家逻辑 (Heuristic AI)。
    """
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """重新开始一局游戏"""
        self.deck = sim.create_deck()
        self.players = [{"hand": [], "melds": [], "fa_count": 0, "is_tenpai": False} for _ in range(4)]
        self.discards_by_player = [[], [], [], []]
        
        # 【核心升级 1】：随机分配庄家 + 随机化底分规则 (Domain Randomization)
        self.dealer_idx = random.randint(0, 3) 
        self.current_player = self.dealer_idx 
        self.current_base_score = random.choice([0, 5, 10, 15])
        self.draw_penalty = self.current_base_score if self.current_base_score > 0 else 5
        
        self.turns = 0
        self.done = False
        self.last_tile = None
        
        # 1. 初始发牌 (13张)
        for _ in range(13):
            for p in self.players:
                p["hand"].append(self.deck.pop())
                
        # 2. 初始补花 (发财)
        for p in self.players:
            while sim.FA in p["hand"]:
                p["hand"].remove(sim.FA)
                p["fa_count"] += 1
                if self.deck:
                    p["hand"].append(self.deck.pop(0)) 
                    
        result = self._player_draw_and_check_turn()
        if isinstance(result, tuple):
            # 如果游戏在初始阶段结束，返回观测张量并标记done
            obs, reward, done, info = result
            self.done = done
            return obs
        else:
            return result
        
    def _player_draw_and_check_turn(self):
        """如果是他家回合，自动模拟直到轮到 Player 0 或结束"""
        while self.deck:
            p = self.players[self.current_player]
            
            # --- 摸牌段 ---
            drawn_tile = self.deck.pop()
            while drawn_tile == sim.FA and self.deck:
                p["fa_count"] += 1
                drawn_tile = self.deck.pop(0)
            
            p["hand"].append(drawn_tile)
            self.last_tile = drawn_tile
            
            # 检查自摸
            s_num = sim.calculate_shanten_accurate(p["hand"], p["melds"])
            if s_num == -1:
                total_p, _ = sim.calculate_final_score(p, drawn_tile, True, {})
                if total_p >= 2:
                    self.done = True
                    return self._finalize_game(winner_idx=self.current_player, score=total_p, is_zimo=True)
            
            # --- 这一步决策权判断 ---
            if self.current_player == 0:
                return self._get_obs()
            
            # --- 他家 (Heuristic AI) 决策弃牌 ---
            discarded_tile = sim.get_best_discard_smart(p["hand"], p["melds"])
            p["hand"].remove(discarded_tile)
            
            # 检查是否有玩家胡牌 (点炮)
            hu_res = self._check_others_hu(self.current_player, discarded_tile)
            if hu_res:
                return hu_res # 游戏结束
                
            # 检查是否有玩家碰/杠
            action_taken = self._check_others_pong_kong(self.current_player, discarded_tile)
            if not action_taken:
                self.discards_by_player[self.current_player].append(discarded_tile)
                self.current_player = (self.current_player + 1) % 4
        
        # 牌抓完了，流局
        self.done = True
        return self._finalize_game(winner_idx=-1, score=0, is_zimo=False)

    def step(self, action):
        """Player 0 执行弃牌动作"""
        if self.done:
            return self._get_obs(), 0, True, {}
        
        p = self.players[0]
        if action not in p["hand"]:
            action = p["hand"][0]
            
        p["hand"].remove(action)
        
        # 1. 检查他家胡这张牌 (点炮)
        hu_res = self._check_others_hu(0, action)
        if hu_res:
            return hu_res
            
        # 2. 检查他家碰/杠
        action_taken = self._check_others_pong_kong(0, action)
        if not action_taken:
            self.discards_by_player[0].append(action)
            self.current_player = 1 # 轮到下家
            
        # 3. 自动运行直到轮到循环回 Player 0
        result = self._player_draw_and_check_turn()
        if isinstance(result, tuple):
            if len(result) == 3:
                return result[0], result[1], result[2], {}
            return result
        else:
            return result, 0, False, {}

    def _check_others_hu(self, current_player_idx, discarded_tile):
        """检查除弃牌者外的所有人是否胡牌"""
        for i in range(1, 4):
            target_idx = (current_player_idx + i) % 4
            tp = self.players[target_idx]
            tp["hand"].append(discarded_tile)
            if sim.calculate_shanten_accurate(tp["hand"], tp["melds"]) == -1:
                score, _ = sim.calculate_final_score(tp, discarded_tile, False, {})
                if score >= 2:
                    self.done = True
                    tp["hand"].remove(discarded_tile)
                    return self._finalize_game(winner_idx=target_idx, score=score, is_zimo=False, loser_idx=current_player_idx)
            tp["hand"].remove(discarded_tile)
        return None

    def _check_others_pong_kong(self, current_player_idx, discarded_tile):
        """检查除弃牌者外的所有人是否碰/杠"""
        for i in range(1, 4):
            target_idx = (current_player_idx + i) % 4
            tp = self.players[target_idx]
            count = tp["hand"].count(discarded_tile)
            if count >= 3: 
                self.last_tile = discarded_tile
                tp["hand"] = [t for t in tp["hand"] if t != discarded_tile]
                tp["melds"].append(("KONG", discarded_tile))
                if self.deck:
                    bonus = self.deck.pop(0)
                    while bonus == sim.FA and self.deck:
                        tp["fa_count"] += 1
                        bonus = self.deck.pop(0)
                    tp["hand"].append(bonus)
                self.current_player = target_idx
                return True
            elif count == 2 and sim.should_pong(tp["hand"], discarded_tile):
                self.last_tile = discarded_tile
                tp["hand"] = [t for t in tp["hand"] if t != discarded_tile]
                tp["melds"].append(("PONG", discarded_tile))
                self.current_player = target_idx
                return True
        return False

    def _get_obs(self):
        """获取 Player 0 的观测张量 (16, 30)"""
        p = self.players[0]
        tot_fa = sum(pl["fa_count"] for pl in self.players)
        opp_melds = []
        for idx in range(1, 4):
            opp_melds.extend(self.players[idx]["melds"])
        rel_discards = [self.discards_by_player[i] for i in range(4)]
        return encode_state(p["hand"], p["melds"], rel_discards, opp_melds, tot_fa, len(self.deck), self.last_tile, 
                           dealer_idx=self.dealer_idx, base_score=self.current_base_score)

    def _calculate_real_money(self, fan, is_dealer_involved):
        """核心换算：current_base_score + ceil(actual_fan / 5) * 5"""
        actual_fan = fan * 2 if is_dealer_involved else fan
        import math
        rounded_fan = math.ceil(actual_fan / 5) * 5 if actual_fan > 0 else 0
        return self.current_base_score + rounded_fan

    def _finalize_game(self, winner_idx, score, is_zimo, loser_idx=None):
        """计算最终真实收益并返回归一化 Reward"""
        obs = self._get_obs()
        net_money = 0
        if winner_idx == -1:
            net_money = (3 * self.draw_penalty) if self.dealer_idx == 0 else -self.draw_penalty
        else:
            is_dealer_hu = (winner_idx == self.dealer_idx)
            if winner_idx == 0:
                if is_zimo:
                    if self.dealer_idx == 0:
                        net_money = 3 * self._calculate_real_money(score, True)
                    else:
                        net_money = self._calculate_real_money(score, True) + 2 * self._calculate_real_money(score, False)
                else:
                    is_loser_dealer = (loser_idx == self.dealer_idx)
                    net_money = self._calculate_real_money(score, is_dealer_hu or is_loser_dealer)
            elif loser_idx == 0:
                is_winner_dealer = (winner_idx == self.dealer_idx)
                net_money = -self._calculate_real_money(score, is_winner_dealer or self.dealer_idx == 0)
            elif is_zimo:
                is_winner_dealer = (winner_idx == self.dealer_idx)
                net_money = -self._calculate_real_money(score, is_winner_dealer or self.dealer_idx == 0)
        reward = np.sign(net_money) * np.sqrt(abs(net_money))
        return obs, reward, True, {}
