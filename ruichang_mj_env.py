import os
import random
import math
import numpy as np
import torch
import ruichang_mj_sim as sim
from state_encoder import encode_state

# 限制 CPU 推理线程数，防止 20 个 Worker 并发抢占导致死锁
torch.set_num_threads(1)

class RuichangMahjongEnv:
    def __init__(self, opponent_ckpt="checkpoints/ppo_elite_v1.pth"):
        self.device = torch.device("cpu")
        self.use_self_play = False
        
        # 延迟导入，防止循环引用
        from resnet_model import MahjongActorCritic
        self.opponent_model = MahjongActorCritic().to(self.device)
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ckpt_path = os.path.join(script_dir, opponent_ckpt)
        
        # 尝试加载上一代的宗师模型作为陪练
        if os.path.exists(ckpt_path):
            try:
                ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                    self.opponent_model.load_state_dict(ckpt['model_state_dict'])
                else:
                    self.opponent_model.load_state_dict(ckpt)
                self.opponent_model.eval()
                self.use_self_play = True
                # 仅在主进程打印一次加载成功的信息，避免 Worker 刷屏
                if multiprocessing.current_process().name == 'MainProcess':
                    print(f"👻 环境加载 Self-Play 陪练成功: {opponent_ckpt}")
            except Exception as e:
                pass
                
        self.reset()

    def reset(self):
        self.deck = sim.create_deck()
        self.players = [{"hand": [], "melds": [], "fa_count": 0, "is_tenpai": False} for _ in range(4)]
        self.discards_by_player = [[], [], [], []]
        
        # 全天候参数化 (Domain Randomization)
        self.dealer_idx = random.randint(0, 3)
        self.current_base_score = random.choice([0, 5, 10, 15])
        
        self.current_player = self.dealer_idx
        self.last_tile = None
        self.done = False
        self.skip_draw = False
        
        # 发牌 & 补花
        for _ in range(13):
            for p in self.players:
                p["hand"].append(self.deck.pop())
                
        for p in self.players:
            while sim.FA in p["hand"]:
                p["hand"].remove(sim.FA)
                p["fa_count"] += 1
                if self.deck:
                    p["hand"].append(self.deck.pop(0))
                    
        return self._player_draw_and_check_turn()

    def _get_obs_for_player(self, p_idx):
        """视角转换器：让任何玩家都以为自己是 Player 0"""
        p = self.players[p_idx]
        tot_fa = sum(pl["fa_count"] for pl in self.players)
        opp_melds = []
        for i in range(1, 4):
            opp_melds.extend(self.players[(p_idx + i) % 4]["melds"])
        
        rel_discards = [
            self.discards_by_player[p_idx],
            self.discards_by_player[(p_idx + 1) % 4],
            self.discards_by_player[(p_idx + 2) % 4],
            self.discards_by_player[(p_idx + 3) % 4]
        ]
        rel_dealer_idx = (self.dealer_idx - p_idx) % 4
        
        return encode_state(
            p["hand"], p["melds"], rel_discards, opp_melds,
            tot_fa, len(self.deck), self.last_tile,
            dealer_idx=rel_dealer_idx,
            base_score=self.current_base_score
        )

    def _get_obs(self):
        return self._get_obs_for_player(0)

    def step(self, action):
        """外部 RL (Player 0) 执行动作"""
        if self.done:
            return self._get_obs(), 0, True, {}
            
        p = self.players[0]
        discarded_tile = action
        if discarded_tile not in p["hand"]:
            discarded_tile = p["hand"][0]
            
        p["hand"].remove(discarded_tile)
        self.last_tile = discarded_tile
        
        # 1. 查点炮
        hu_res = self._check_others_hu(0, discarded_tile)
        if hu_res: return hu_res
        
        # 2. 查碰杠
        action_taken = self._check_others_pong_kong(0, discarded_tile)
        if not action_taken:
            self.discards_by_player[0].append(discarded_tile)
            self.current_player = 1
            self.skip_draw = False
            
        return self._player_draw_and_check_turn()

    def _player_draw_and_check_turn(self):
        """环境内部状态机推进"""
        while not self.done and self.deck:
            if self.current_player == 0 and not self.skip_draw:
                # 轮到主模型 (Player 0) 摸牌并决策
                p = self.players[0]
                drawn_tile = self.deck.pop()
                while drawn_tile == sim.FA and self.deck:
                    p["fa_count"] += 1
                    drawn_tile = self.deck.pop(0)
                    
                p["hand"].append(drawn_tile)
                self.last_tile = drawn_tile
                
                if sim.calculate_shanten_accurate(p["hand"], p["melds"]) == -1:
                    fan, _ = sim.calculate_final_score(p, drawn_tile, True, {})
                    if fan >= 2:
                        self.done = True
                        return self._finalize_game(0, fan, True)
                        
                return self._get_obs(), 0, False, {}
                
            if self.current_player != 0:
                p = self.players[self.current_player]
                
                # NPC 摸牌
                if not self.skip_draw:
                    drawn_tile = self.deck.pop()
                    while drawn_tile == sim.FA and self.deck:
                        p["fa_count"] += 1
                        if not self.deck: break
                        drawn_tile = self.deck.pop(0)
                        
                    p["hand"].append(drawn_tile)
                    self.last_tile = drawn_tile
                    
                    if sim.calculate_shanten_accurate(p["hand"], p["melds"]) == -1:
                        fan, _ = sim.calculate_final_score(p, drawn_tile, True, {})
                        if fan >= 2:
                            self.done = True
                            return self._finalize_game(self.current_player, fan, True)
                
                self.skip_draw = False
                
                # ==========================================
                # 非对称 Self-Play 混合决策矩阵
                # ==========================================
                discarded_tile = None
                if self.current_player == 1 or not self.use_self_play:
                    # Player 1: 查表专家 (维持底层压力与偶然性)
                    discarded_tile = sim.get_best_discard_smart(p["hand"], p["melds"])
                else:
                    # Player 2 & 3: 神经网络陪练 (使用独立加载的模型前向推断)
                    obs_tensor = self._get_obs_for_player(self.current_player)
                    obs_t = torch.from_numpy(obs_tensor).unsqueeze(0).to(self.device).float()
                    mask_t = torch.from_numpy(obs_tensor[13]).unsqueeze(0).to(self.device).float()
                    
                    with torch.no_grad():
                        logits, _ = self.opponent_model(obs_t, legal_mask=mask_t)
                        # 加入轻微温度系数避免防守死板
                        probs = torch.nn.functional.softmax(logits / 0.05, dim=-1)
                        discarded_tile = torch.multinomial(probs, 1).squeeze(-1).item()
                        
                    if discarded_tile not in p["hand"]: discarded_tile = p["hand"][0]

                p["hand"].remove(discarded_tile)
                self.last_tile = discarded_tile
                
                hu_res = self._check_others_hu(self.current_player, discarded_tile)
                if hu_res: return hu_res
                
                action_taken = self._check_others_pong_kong(self.current_player, discarded_tile)
                if not action_taken:
                    self.discards_by_player[self.current_player].append(discarded_tile)
                    self.current_player = (self.current_player + 1) % 4
            else:
                # 触发此条件说明 Player 0 碰/杠了牌 (skip_draw=True)，挂起环境交还控制权
                return self._get_obs(), 0, False, {}

        # 牌山摸完，流局
        if not self.done:
            self.done = True
            return self._finalize_game(-1, 0, False)

    def _check_others_hu(self, current_player, discarded_tile):
        """查点炮"""
        for i in range(1, 4):
            target_idx = (current_player + i) % 4
            tp = self.players[target_idx]
            tp["hand"].append(discarded_tile)
            if sim.calculate_shanten_accurate(tp["hand"], tp["melds"]) == -1:
                fan, _ = sim.calculate_final_score(tp, discarded_tile, False, {})
                if fan >= 2:
                    self.done = True
                    return self._finalize_game(target_idx, fan, False, loser_idx=current_player)
            tp["hand"].remove(discarded_tile)
        return None

    def _check_others_pong_kong(self, current_player, discarded_tile):
        """查碰杠 (Player 0 仅被动执行)"""
        for i in range(1, 4):
            target_idx = (current_player + i) % 4
            tp = self.players[target_idx]
            count = tp["hand"].count(discarded_tile)
            
            if count >= 3:
                tp["hand"] = [t for t in tp["hand"] if t != discarded_tile]
                tp["melds"].append(("KONG", discarded_tile))
                if self.deck:
                    bonus = self.deck.pop(0)
                    while bonus == sim.FA and self.deck:
                        tp["fa_count"] += 1
                        if not self.deck: break
                        bonus = self.deck.pop(0)
                    if bonus is not None: tp["hand"].append(bonus)
                self.current_player = target_idx
                self.skip_draw = True
                return True
                
            elif count == 2 and sim.should_pong(tp["hand"], discarded_tile):
                tp["hand"] = [t for t in tp["hand"] if t != discarded_tile]
                tp["melds"].append(("PONG", discarded_tile))
                self.current_player = target_idx
                self.skip_draw = True
                return True
        return False

    # ==========================================
    # 💰 核心计分公式引擎：完美对齐真实金币流转
    # ==========================================
    def _calculate_real_money(self, fan, is_dealer_involved):
        """
        算法：底分 + 取整分
        1. 庄家参与则番数翻倍
        2. 将有效番数按 5 向上取整
        """
        effective_fan = fan * 2 if is_dealer_involved else fan
        points = math.ceil(effective_fan / 5.0) * 5
        return self.current_base_score + points

    def _finalize_game(self, winner_idx, fan, is_zimo, loser_idx=None):
        obs = self._get_obs()
        net_money = 0
        
        if winner_idx == -1:
            # 【惩罚机制】：流局且未听牌，扣除大额惩罚，逼迫 AI 保持进攻阵型
            shanten = sim.calculate_shanten_accurate(self.players[0]["hand"], self.players[0]["melds"])
            if shanten > 0:
                net_money -= (self.current_base_score + 10)
        else:
            if winner_idx == 0:
                # RL (Player 0) 赢牌
                if is_zimo:
                    # 自摸：收三家钱，分别计算是否涉及庄家
                    for i in range(1, 4):
                        is_dealer = (self.dealer_idx == 0 or self.dealer_idx == i)
                        net_money += self._calculate_real_money(fan, is_dealer)
                else:
                    # 点炮：只收点炮者的钱
                    is_dealer = (self.dealer_idx == 0 or self.dealer_idx == loser_idx)
                    net_money += self._calculate_real_money(fan, is_dealer)
            else:
                # RL (Player 0) 输牌
                if is_zimo:
                    # 别人自摸：RL 赔一份
                    is_dealer = (self.dealer_idx == 0 or self.dealer_idx == winner_idx)
                    net_money -= self._calculate_real_money(fan, is_dealer)
                elif loser_idx == 0:
                    # RL 点炮给别人：RL 赔一份
                    is_dealer = (self.dealer_idx == 0 or self.dealer_idx == winner_idx)
                    net_money -= self._calculate_real_money(fan, is_dealer)

        # 奖励塑形：开方平滑缩放，防止 100 炮极值梯度爆炸，同时保留正负号
        reward = math.copysign(math.sqrt(abs(net_money)), net_money)
        return obs, reward, True, {}