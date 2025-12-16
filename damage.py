import random
import numpy as np
from scipy.stats import norm


SIGNIFICANCE_LEVEL = 0.001 # 显著性水平。如果 P值 < SIGNIFICANCE_LEVEL，则认为公式不合理。


"""
进行数学期望计算时，假设勇士的生命值为正无穷大（原因在知乎文章贤者之证部分有具体论述）。
贤者之证的大回复不考虑。
贤者之证的集能改成集能时立刻结算，也不影响计算结果。


贤者之证
1级:功效为:回血:当怪物的一次攻击没打中主角时,主角有61%的几率回复体力,回复体力=怪物攻击力/5。
2级:功效为:回血:当怪物的一次攻击没打中主角时,主角有81%的几率回复体力,回复体力=怪物攻击力/5。集能:当非魔法师系怪物的一次攻击击中主角且能造成伤害时,主角有11%的几率收集一次能量,同时减伤50点(怪物暴击判定前),每次收集的能量会使主角在战斗胜利后随机回复0到29点体力。
3级:功效为:回血:当怪物的一次攻击没打中主角时,回复体力=怪物攻击力/5。集能:当怪物的一次攻击击中主角时,如果该怪物不是魔法师系怪物且能对主角造成伤害,那么主角有31%的几率收集一次能量,同时减伤50点(怪物暴击判定前),每次收集的能量会使主角在最近的一次战斗胜利后随机回复0到29点体力;如果该怪物是魔法师系怪物,那么主角有21%的几率收集一次能量,同时减伤100点(怪物暴击判定前),每次收集的能量会使主角在战斗胜利后随机回复体力为:怪物攻击力到怪物攻击力+19。
(所有减伤均可减为负数)

霸者之证
1级:功效为:反弹:当非魔法师系怪物的一次攻击击中主角时,主角有6%的几率反弹攻击,如果怪物对主角的伤害/2≧怪物体力,那么怪物直接死亡,否则主角承受伤害=怪物对主角的伤害,怪物承受伤害=怪物对主角的伤害/2。
2级:功效为:反弹:当非魔法师系怪物的一次攻击击中主角时,主角有11%的几率反弹攻击,如果怪物对主角的伤害≧怪物体力,那么怪物直接死亡,否则主角承受伤害=怪物对主角的伤害/2,怪物承受伤害=怪物对主角的伤害。霸体:当非魔法师系怪物的一次攻击击中主角但没有触发反弹功效时,主角有11%的几率本次攻击伤害计算中防御力*2。
3级:功效为:反弹:当非魔法师系怪物的一次攻击击中主角时,主角有16%的几率反弹攻击,如果怪物攻击力/2≧怪物体力,那么怪物直接死亡,否则主角承受伤害=怪物对主角的伤害/3,怪物承受伤害=怪物攻击力/3。霸体:当非魔法师系怪物的一次攻击击中主角但没有触发反弹功效时,主角有16%的几率本次攻击伤害计算中防御力*2。破防:当主角的一次攻击击中怪物时,怪物有11%的几率本次攻击伤害计算中防御力-怪物防御力/1.3。
(攻击反弹时,怪物不会暴击。当怪物击中主角时,先进行是否反弹的判定,如果判定为不反弹,接着才会进入暴击的判定。)

勇者之证
1级:功效为:无
2级:功效为:攻击次数+1。(勇士每回合可以进行2次独立的攻击)
3级:功效为:当主角的一次攻击击中怪物且能造成伤害,并且主角防御力＜怪物攻击力时,有16%的几率主角对怪物造成附加伤害(暴击判定前),数值为主角的防御力/4。 (对坚固怪无效)
(勇者之证3级时2级效果同样成立)


异常状态：
有些怪物每次击中勇士时，会有一个固定的概率使勇士进入异常状态（中毒，衰弱），即使这次攻击伤害是0也会判定，但如果被勇士闪避则不会判定。
霸者之证的反弹生效时，也会进行这个判定。即使这个反弹把怪物杀死了，这次反弹也会进行这个判定。
这个异常状态不会影响战斗，在战斗后才开始起效。
魔攻的怪物无法让勇士进入异常状态。
本文件包含了一个函数，用于计算一次战斗后勇士进入异常状态的概率。

秒杀：
当怪物不是魔法师，怪物的攻击击中了勇士，没有触发霸者之证的反弹或者霸体效果，且怪物的攻击力与勇士的防御临界之和小于勇士的防御（即怪物不能对勇士造成伤害），
且勇士攻击力大于怪物防御力时，有16%概率触发秒杀效果，立刻终止战斗，怪物直接被杀死。
秒杀的这一回合不会判定异常状态。


"""

class MagicTowerSimulator:
    def _round_half_up(self, value):
            """
            Helper function to implement standard mathematical rounding (round half up).
            执行四舍五入到整数。
            """
            return int(value + 0.5 + 1e-8)

    def __init__(self, 
                 m_hp, m_atk, m_def, m_eva, m_crit,  # m=怪物的,hp=生命值,atk=攻击力,def=防御力,eva=闪避率,crit=暴击率
                 h_atk, h_def, h_eva, h_crit,        # h=勇士的
                 h_atk_thresh=0, h_def_thresh=0,     # 攻击临界值，防御临界值
                 special_type=None,  # 怪物的特殊能力，只允许 'solid', 'mimic', 'magic', 'k_combo' ，None
                 k_value=1,          # 怪物的连击次数，仅当 special_type='k_combo' 时有效
                 emblem_type=None,   # 章的类型 (None, 'sage', 'hero', 'overlord')
                 emblem_level=0,     # 章的等级 (0-3)
                 abnormal_prob=0     # 怪物击中勇士时使勇士进入异常状态的概率
                 ):   
        """
        初始化并校验数值。
        """
        # =========================
        # 1. 输入有效性检查 (Validation)
        # =========================
        valid_types = {None, 'solid', 'mimic', 'magic', 'k_combo'}
        if special_type not in valid_types:
            raise ValueError(f"Invalid special_type: '{special_type}'. Must be one of {valid_types}")

        if not isinstance(m_hp, int) or m_hp <= 0:
            raise ValueError(f"Monster HP must be a positive integer.")

        if not isinstance(k_value, int) or k_value <= 0:
            raise ValueError(f"Combo count (k_value) must be a positive integer.")

        non_negative_ints = {
            "m_atk": m_atk, "m_def": m_def, 
            "h_atk": h_atk, "h_def": h_def,
            "h_atk_thresh": h_atk_thresh, "h_def_thresh": h_def_thresh
        }
        for name, val in non_negative_ints.items():
            if not isinstance(val, int) or val < 0:
                raise ValueError(f"Attribute '{name}' must be a non-negative integer.")

        probs = {
            "m_eva": m_eva, "m_crit": m_crit,
            "h_eva": h_eva, "h_crit": h_crit, "abnormal_prob": abnormal_prob
        }
        for name, val in probs.items():
            if not isinstance(val, (int, float)) or not (0 <= val <= 1):
                raise ValueError(f"Probability '{name}' must be a float between 0 and 1.")
            
        if m_eva > 0.991 or h_eva > 0.991:
            raise ValueError("Evasion rates must be strictly less than 1 to avoid infinite loops.")

        valid_emblems = {None, 'sage', 'hero', 'overlord'}
        if emblem_type not in valid_emblems:
            raise ValueError(f"Invalid emblem_type: '{emblem_type}'. Must be one of {valid_emblems}")
        
        if not isinstance(emblem_level, int) or not (0 <= emblem_level <= 3):
            raise ValueError(f"Emblem level must be an integer between 0 and 3. Got: {emblem_level}")
        
        if emblem_type is not None and emblem_level == 0:
            raise ValueError("Emblem level must be at least 1 if emblem_type is specified.")
        
        # =========================
        # 2. 属性赋值
        # =========================

        self.emblem_type = emblem_type
        self.emblem_level = emblem_level
        self.m_hp_max = m_hp
        self.m_atk_raw = m_atk
        self.m_def_raw = m_def
        self.m_eva = float(m_eva)
        self.m_crit = float(m_crit)
        
        self.h_atk = h_atk
        self.h_def = h_def
        self.h_eva = float(h_eva)
        self.h_crit = float(h_crit)
        
        self.h_atk_thresh = h_atk_thresh
        self.h_def_thresh = h_def_thresh
        
        self.special_type = special_type
        self.k_value = k_value if special_type == 'k_combo' else 1
        if self.special_type == 'magic':
            abnormal_prob = 0
        self.have_abnormal = abnormal_prob > 1e-6
        self.abnormal_prob = float(abnormal_prob)


        # 战前属性调整 (坚固/仿攻)
        self.m_def = self.m_def_raw
        if self.special_type == 'solid':
            if self.h_atk > self.m_def_raw:
                self.m_def = self.h_atk
                
        self.m_atk = self.m_atk_raw
        if self.special_type == 'mimic':
            if self.h_atk > self.m_atk_raw:
                self.m_atk = self.h_atk

    def _calculate_base_damage(self, atk, defense, threshold):
        """
        计算一次普通攻击造成的伤害，这个函数必须返回非负整数
        """
        assert isinstance(atk, int) and isinstance(defense, int) and isinstance(threshold, int)
        if atk > defense:
            return atk - defense
        elif defense - threshold <= atk <= defense:
            return 1
        else:
            return 0

    def simulate_once(self, return_abnormal=False):
        """
        模拟一次完整的战斗过程,返回勇士受到的总伤害。
        如果打开return_abnormal，返回0表示不会进入异常状态，1表示会进入异常状态。
        """
        
        if return_abnormal and ((not self.have_abnormal) or self.special_type == 'magic'):
            return 0

        is_ol_3 = self.emblem_type == 'overlord' and self.emblem_level == 3
        is_non_magic = self.special_type != 'magic'

        # ==================================================
        # NEW UNWINNABLE CHECK: 确保至少有一种方式能对怪物造成伤害
        # ==================================================
        
        # 1. 勇士标准攻击伤害潜力
        max_potential_hero_dmg = self._calculate_base_damage(self.h_atk, self.m_def, self.h_atk_thresh)
        
        # 2. 勇士破防 (Breach) 攻击伤害潜力
        if is_ol_3:
            # 计算最大的防御减少量(即破防成功)
            reduction_max = self._round_half_up(self.m_def / 1.3)
            m_def_min = max(0, self.m_def - reduction_max)
            
            # 计算破防成功时的伤害
            breach_dmg = self._calculate_base_damage(self.h_atk, m_def_min, self.h_atk_thresh)
            
            # 取标准攻击伤害和破防伤害中的最大值
            max_potential_hero_dmg = max(max_potential_hero_dmg, breach_dmg)
            
        # 3. 反弹 (Counter-Attack) 伤害潜力
        # 根据不同等级的霸者之证，反弹时怪物承受伤害的计算规则不同：
        #  - 等级1: 怪物承受 原始基础伤害 / 2（四舍五入）
        #  - 等级2: 怪物承受 原始基础伤害
        #  - 等级3: 怪物承受 怪物攻击力 / 3（四舍五入）  
        counter_dmg_potential = 0
        if self.emblem_type == 'overlord' and is_non_magic:
            original_dmg_base = self._calculate_base_damage(self.m_atk, self.h_def, self.h_def_thresh)

            if self.emblem_level == 1:
                counter_dmg_potential = self._round_half_up(original_dmg_base / 2)
            elif self.emblem_level == 2:
                counter_dmg_potential = original_dmg_base
            else:
                # level 3
                counter_dmg_potential = self._round_half_up(self.m_atk / 3)

        # 最终判定：如果所有伤害来源都不能对怪物造成伤害，则战斗无法胜利 (np.nan)
        if max_potential_hero_dmg <= 0 and counter_dmg_potential <= 0:
             return np.nan
        
        # ==================================================
        # END NEW UNWINNABLE CHECK
        # ==================================================
        
        current_m_hp = self.m_hp_max
        total_hero_damage_taken = 0
        
        # 预设怪物攻击阶段的初始防御力(处理魔攻)
        h_def = 0 if self.special_type == 'magic' else self.h_def
        
        while True:
            # =========================
            # 1. 勇士发起攻击 (支持勇者徽章多次攻击与3级附加伤害)
            # =========================
            hero_attacks = 1
            if self.emblem_type == 'hero' and self.emblem_level >= 2:
                hero_attacks = 2

            for _atk in range(hero_attacks):
                modified_m_def = self.m_def

                # A. 判定怪物闪避
                if random.random() < self.m_eva:
                    damage_to_monster = 0
                else:
                    # B. 命中：检查霸者3级破防 (Breach - 11%)
                    if is_ol_3 and random.random() < 0.11:
                        # 怪物防御力/1.3 四舍五入到整数
                        reduction = self._round_half_up(self.m_def / 1.3)
                        modified_m_def = self.m_def - reduction

                    # C. 使用可能修正的防御力计算基础伤害
                    current_hero_base_dmg = self._calculate_base_damage(self.h_atk, modified_m_def, self.h_atk_thresh)

                    # 勇者之证3级的附加伤害（暴击判定前），在非坚固怪且满足防御条件时触发
                    extra_hero_dmg = 0
                    if (self.emblem_type == 'hero' and self.emblem_level == 3
                            and self.h_def < self.m_atk and self.special_type != 'solid'):
                        if current_hero_base_dmg > 0 and random.random() < 0.16:
                            extra_hero_dmg = self._round_half_up(self.h_def / 4)

                    # D. 判定勇士暴击 (附加伤害计入暴击倍数)
                    if random.random() < self.h_crit:
                        damage_to_monster = (current_hero_base_dmg + extra_hero_dmg) * 2
                    else:
                        damage_to_monster = current_hero_base_dmg + extra_hero_dmg

                current_m_hp -= damage_to_monster

                # 判定怪物死亡
                if current_m_hp <= 0:
                    break

            # 判定怪物死亡
            if current_m_hp <= 0:
                break
            
            # =========================
            # 2. 怪物发起攻击
            # =========================
            
            # 怪物 K 连击循环
            for _ in range(self.k_value):
                # 每次攻击的有效防御力，初始为 h_def (已处理魔攻)
                current_h_def_for_calc = h_def 

                # Evasion Check
                if random.random() < self.h_eva:
                    damage_to_hero = 0
                    # 贤者之证：怪物未命中时可能触发回血
                    if self.emblem_type == 'sage':
                        if self.emblem_level == 1:
                            if random.random() < 0.61:
                                heal = self._round_half_up(self.m_atk / 5)
                                total_hero_damage_taken -= heal
                        elif self.emblem_level == 2:
                            if random.random() < 0.81:
                                heal = self._round_half_up(self.m_atk / 5)
                                total_hero_damage_taken -= heal
                        else:
                            # 等级3：始终回血
                            heal = self._round_half_up(self.m_atk / 5)
                            total_hero_damage_taken -= heal
                else:
                    # --- 命中：根据徽章类型处理额外效果（霸者/贤者） ---
                    # 先处理霸者之证反弹/霸体（如存在）
                    have_counter = False
                    have_over_body = False
                    if is_non_magic and self.emblem_type == 'overlord':
                        # 设置不同等级的反弹与霸体概率
                        if self.emblem_level == 1:
                            counter_prob = 0.06
                            over_body_prob = 0.0
                        elif self.emblem_level == 2:
                            counter_prob = 0.11
                            over_body_prob = 0.11
                        else:
                            counter_prob = 0.16
                            over_body_prob = 0.16

                        # 反弹判定（优先）
                        if random.random() < counter_prob:
                            have_counter = True
                            hero_takes = 0
                            monster_takes = 0
                            original_dmg_base = self._calculate_base_damage(self.m_atk, h_def, self.h_def_thresh)
                            if self.emblem_level == 1:
                                if self._round_half_up(original_dmg_base / 2) >= current_m_hp:
                                    current_m_hp = 0
                                    # 注意这个地方不能break，因为在后面有判断异常状态这一步仍需执行
                                else:
                                    hero_takes = original_dmg_base
                                    monster_takes = self._round_half_up(original_dmg_base / 2)
                            elif self.emblem_level == 2:
                                if original_dmg_base >= current_m_hp:
                                    current_m_hp = 0
                                else:
                                    hero_takes = self._round_half_up(original_dmg_base / 2)
                                    monster_takes = original_dmg_base
                            else:
                                if self._round_half_up(self.m_atk / 2) >= current_m_hp:
                                    current_m_hp = 0
                                else:
                                    hero_takes = self._round_half_up(original_dmg_base / 3)
                                    monster_takes = self._round_half_up(self.m_atk / 3)

                            total_hero_damage_taken += hero_takes
                            current_m_hp -= monster_takes

                        # 霸体判定
                        if over_body_prob > 0 and random.random() < over_body_prob:
                            have_over_body = True
                            current_h_def_for_calc *= 2

                    if not have_counter:
                        # 贤者之证：命中且能造成伤害时可能触发集能（在暴击判定前减伤）
                        original_dmg_base = self._calculate_base_damage(self.m_atk, current_h_def_for_calc, self.h_def_thresh)
                        dmg_reduction = 0
                        # `heal_now` is the post-battle heal amount that should NOT
                        # participate in crit calculation. It will be applied
                        # separately to `total_hero_damage_taken` after the damage
                        # for this hit is computed.
                        heal_now = 0
                        if self.emblem_type == 'sage' and original_dmg_base > 0:
                            # When energy (集能) triggers, the fixed damage
                            # reduction (50/100) applies before crit. The random
                            # post-battle heal is produced now but applied
                            # separately so it doesn't affect crit.
                            if self.emblem_level == 2 and is_non_magic:
                                if random.random() < 0.11:
                                    dmg_reduction = 50
                                    heal_now = random.randint(0, 29)
                            elif self.emblem_level == 3:
                                if not self.special_type == 'magic':
                                    if random.random() < 0.31:
                                        dmg_reduction = 50
                                        heal_now = random.randint(0, 29)
                                else:
                                    if random.random() < 0.21:
                                        dmg_reduction = 100
                                        heal_now = random.randint(self.m_atk, self.m_atk + 19)

                        # --- 标准伤害计算（应用贤者减伤） ---
                        monster_base_dmg = original_dmg_base - dmg_reduction
                        # 判定怪物暴击
                        if random.random() < self.m_crit:
                            damage_to_hero = monster_base_dmg * 2
                        else:
                            damage_to_hero = monster_base_dmg
                        total_hero_damage_taken += damage_to_hero
                        # Apply the post-battle heal (converted to immediate heal)
                        # separately so it does not affect crit calculation. This
                        # may reduce total damage taken (can be negative effect).
                        if heal_now:
                            total_hero_damage_taken -= heal_now

                        # 秒杀效果判定
                        if self.special_type!='magic' and original_dmg_base==0 and not have_over_body and self.h_atk > self.m_def:
                            if random.random() < 0.16:
                                current_m_hp = 0
                                break
                
                    # 判定异常状态
                    # 秒杀时将会直接终止整个战斗流程，本回合不会再判定异常状态。
                    # 但如果霸者之证的反弹在本回合杀死了怪物，则本回合依然会判定异常状态。
                    # 新新魔塔V1.1的原代码如此。
                    if return_abnormal and self.have_abnormal and random.random() < self.abnormal_prob:
                        return 1

                # 判定怪物死亡
                if current_m_hp <= 0:
                    break

            # 判定怪物死亡
            if current_m_hp <= 0:
                break

        if return_abnormal:
            return 0

        return total_hero_damage_taken

    def monte_carlo_simulation(self, n_trials=10000):
        """
        运行多次模拟，计算详细统计指标。
        返回一个包含统计数据的字典。
        """
        results = []
        
        # 批量运行模拟
        for _ in range(n_trials):
            res = self.simulate_once()
            # 如果出现 NaN (打不过)，则整体统计无效，直接返回 NaN 标记
            if np.isnan(res):
                return {
                    "count": n_trials,
                    "mean": np.nan,
                    "median": np.nan,
                    "max": np.nan,
                    "min": np.nan,
                    "std": np.nan
                }
            results.append(res)
            
        # 转换为 NumPy 数组以便快速计算统计量
        results_array = np.array(results)
        
        stats = {
            "count": n_trials,
            "mean": np.mean(results_array),
            "median": np.median(results_array),
            "max": np.max(results_array),
            "min": np.min(results_array),
            "std": np.std(results_array)
        }
        
        return stats

    def monte_carlo_simulation_abnormal(self, n_trials=10000):
        """
        运行多次模拟，计算进入异常状态的次数。
        返回一个包含统计数据的字典。
        """        
        if not self.have_abnormal:
            return {
                "count": n_trials,
                "abnormal_count": 0,
                "abnormal_rate": 0.0
            }
        
        abnormal_count = 0
        
        # 批量运行模拟
        for _ in range(n_trials):
            res = self.simulate_once(return_abnormal=True)
            if np.isnan(res):
                return {
                    "count": n_trials,
                    "abnormal_count": np.nan,
                    "abnormal_rate": np.nan
                }
            abnormal_count += res  # res is 0 or 1
            
        abnormal_rate = abnormal_count / n_trials
        
        stats = {
            "count": n_trials,
            "abnormal_count": abnormal_count,
            "abnormal_rate": abnormal_rate
        }
        
        return stats


    def calculate_formula_expectation(self):
        """
        计算期望伤害精确值的公式解。
        """
        H = self.m_hp_max
        p, q = self.m_eva, self.m_crit
        u, v = self.h_eva, self.h_crit
        K = self.k_value
        
        if self.emblem_type is None or (self.emblem_type == 'hero' and self.emblem_level == 1):
            hero_base = self._calculate_base_damage(self.h_atk, self.m_def, self.h_atk_thresh)
            if hero_base <= 0:
                return np.nan
                
            M = np.ceil(H / hero_base)
            effective_h_def = 0 if self.special_type == 'magic' else self.h_def
            monster_base = self._calculate_base_damage(self.m_atk, effective_h_def, self.h_def_thresh)
            monster_dmg_per_turn = self.k_value * monster_base * (1 - u) * (1 + q)
            
            expected_turns = 1 / ((1 - p) * ((1 + v) ** 2)) * (M * (1 + v) + v * (1 - (-v) ** M)) - 1
            
            return monster_dmg_per_turn * expected_turns
        elif self.emblem_type == 'overlord':
            lvl = self.emblem_level
            # 勇士对怪物的进攻手段
            qq = np.zeros(5)
            mm = np.zeros(5, dtype=np.int64)
            if lvl == 3:
                # 怪物闪避
                qq[0] = p
                mm[0] = 0
                # 普攻且不触发破防
                qq[1] = (1-p) * (1-v) * (1-0.11)
                mm[1] = self._calculate_base_damage(self.h_atk, self.m_def, self.h_atk_thresh) 
                # 普攻且触发破防
                qq[2] = (1-p) * (1-v) * 0.11
                reduction = self._round_half_up(self.m_def / 1.3)
                mm[2] = self._calculate_base_damage(self.h_atk, self.m_def - reduction, self.h_atk_thresh)
                # 暴击且不触发破防
                qq[3] = (1-p) * v * (1-0.11)
                mm[3] = self._calculate_base_damage(self.h_atk, self.m_def, self.h_atk_thresh) * 2
                # 暴击且触发破防
                qq[4] = (1-p) * v * 0.11
                mm[4] = self._calculate_base_damage(self.h_atk, self.m_def - reduction, self.h_atk_thresh) * 2
            else:
                # 怪物闪避
                qq[0] = p
                mm[0] = 0
                # 普攻
                qq[1] = (1-p) * (1-v)
                mm[1] = self._calculate_base_damage(self.h_atk, self.m_def, self.h_atk_thresh) 
                # 暴击
                qq[2] = (1-p) * v
                mm[2] = self._calculate_base_damage(self.h_atk, self.m_def, self.h_atk_thresh) * 2

            
            # mm必须全部是非负整数
            if not all(isinstance(x, np.int64) and x >= 0 for x in mm):
                print(mm)
                raise Exception('Damage values mm must be non-negative integers in formula calculation.')
            
            # Determine counter / over_body params per level
            if lvl == 1:
                counter_prob = 0.06
                over_body_prob = 0.0
                original_dmg_base = self._calculate_base_damage(self.m_atk, self.h_def, self.h_def_thresh)
                counter = self._round_half_up(original_dmg_base / 2)  # monster takes
                counter_big = counter
                nn_counter = original_dmg_base  # hero takes when counter
            elif lvl == 2:
                counter_prob = 0.11
                over_body_prob = 0.11
                original_dmg_base = self._calculate_base_damage(self.m_atk, self.h_def, self.h_def_thresh)
                counter = original_dmg_base
                counter_big = original_dmg_base
                nn_counter = self._round_half_up(original_dmg_base / 2)
            else:
                counter_prob = 0.16
                over_body_prob = 0.16
                counter = self._round_half_up(self.m_atk / 3)
                counter_big = self._round_half_up(self.m_atk / 2)
                original_dmg_base = self._calculate_base_damage(self.m_atk, self.h_def, self.h_def_thresh)
                nn_counter = self._round_half_up(original_dmg_base / 3)

            # 无法战斗判定
            if all(x == 0 for x in mm) and (counter == 0 or self.special_type == 'magic'):
                return np.nan   

            if self.special_type != 'magic':           
                # 怪物对勇士的进攻手段
                pp = np.zeros(6)
                nn = np.zeros(6)
                # 触发闪避
                pp[0] = u
                nn[0] = 0
                # 触发反弹
                pp[1] = (1-u) * counter_prob
                nn[1] = nn_counter     
                # 普通攻击不触发霸体
                pp[2] = (1-u) * (1-q) * (1-counter_prob) * (1-over_body_prob)
                nn[2] = self._calculate_base_damage(self.m_atk, self.h_def, self.h_def_thresh)
                # 普通攻击触发霸体
                pp[3] = (1-u) * (1-q) * (1-counter_prob) * over_body_prob
                nn[3] = self._calculate_base_damage(self.m_atk, self.h_def * 2, self.h_def_thresh)
                # 暴击攻击不触发霸体
                pp[4] = (1-u) * q * (1-counter_prob) * (1-over_body_prob)
                nn[4] = self._calculate_base_damage(self.m_atk, self.h_def, self.h_def_thresh) * 2
                # 暴击攻击触发霸体
                pp[5] = (1-u) * q * (1-counter_prob) * over_body_prob
                nn[5] = self._calculate_base_damage(self.m_atk, self.h_def * 2, self.h_def_thresh) * 2
                # 未触发反弹时的期望伤害
                E_dmg_no_counter = sum(pp[i] * nn[i] for i in [0,2,3,4,5]) / (1-pp[1])
                # 触发反弹时的期望伤害
                E_dmg_counter = nn[1]

                # 计算回合数期望递推公式    
                # F[0][x]表示怪物血量为x，下一步为勇者对怪物发起攻击时，直到怪物死亡，触发反弹次数的数学期望（不计最后一回合）。
                # F[i][x]表示怪物血量为x，下一步为怪物对勇者发起第i次攻击时，直到怪物死亡，触发反弹次数的数学期望（不计最后一回合）。
                # G[0][x]表示怪物血量为x，下一步为勇者对怪物发起攻击时，直到怪物死亡，怪物对勇士未被反弹的攻击次数的数学期望（不计最后一回合）。
                # G[i][x]表示怪物血量为x，下一步为怪物对勇士发起第i次攻击时，直到怪物死亡，怪物对勇士未被反弹的攻击次数的数学期望（不计最后一回合）。

                # F[0][x] = sum( qq[i] * F[1][x - mm[i]] )  (0<= i <=4)
                # F[i][x] = pp[1]*(F[i+1][x-counter]+1)*indicator(x>counter_big) + (1-pp[1])*F[i+1][x]   (1<= i <= K)(F[K+1]=F[0])
                # G[0][x] = sum( qq[i] * G[1][x - mm[i]] )  (0<= i <=4)
                # G[i][x] = pp[1]*(G[i+1][x-counter])*indicator(x>counter_big) + (1-pp[1])*(G[i+1][x]+1)   (1<= i <= K)(G[K+1]=G[0])
                F = np.zeros((K+2, H+1))
                G = np.zeros((K+2, H+1))
                for h in range(1, H+1):
                    # 计算F[0][h]
                    # 维护表达式：F[0][h] = A[i] * F[i][h] + B[i]    (1<= i <= K+1) 直到循环回到 F[0][h]
                    A = np.zeros(K+2)
                    B = np.zeros(K+2)
                    for j in range(5):
                        if mm[j]==0:
                            A[1] += qq[j]
                        else:
                            B[1] += qq[j] * F[1][max(0, h - mm[j])]
                    for i in range(1, K+1):
                        if h > counter_big:
                            if counter==0:
                                A[i+1] += pp[1] * A[i]
                                B[i+1] += pp[1] * A[i]
                            else: 
                                B[i+1] += pp[1] * A[i] * (F[i+1][h - counter] + 1)
                        A[i+1] += (1 - pp[1]) * A[i]
                        B[i+1] += B[i]
                    assert(A[K+1] < 1 - 1e-6)
                    F[0][h] = B[K+1] / (1 - A[K+1])
                    F[K+1][h] = F[0][h]
                    # 计算F[i][h] (1<= i <= K)
                    for i in range(K, 0,-1):
                        if h > counter_big:
                            F[i][h] = pp[1] * (F[i+1][h - counter] + 1) + (1 - pp[1]) * F[i+1][h]
                        else:
                            F[i][h] = (1 - pp[1]) * F[i+1][h]
                    # 计算G[0][h]
                    # 维护表达式：G[0][h] = C[i] * G[i][h] + D[i]    (1<= i <= K+1) 直到循环回到 G[0][h]
                    C = np.zeros(K+2)
                    D = np.zeros(K+2)
                    for j in range(5):
                        if mm[j]==0:
                            C[1] += qq[j]
                        else:
                            D[1] += qq[j] * G[1][max(0, h - mm[j])]
                    for i in range(1, K+1):
                        if h > counter_big:
                            if counter==0:
                                C[i+1] += pp[1] * C[i]
                            else:
                                D[i+1] += pp[1] * C[i] * G[i+1][h - counter]
                        C[i+1] += (1 - pp[1]) * C[i]
                        D[i+1] += D[i] + (1 - pp[1]) * C[i]
                    assert(C[K+1] < 1 - 1e-6)
                    G[0][h] = D[K+1] / (1 - C[K+1])
                    G[K+1][h] = G[0][h]
                    # 计算G[i][h] (1<= i <= K)
                    for i in range(K, 0,-1):
                        if h > counter_big:
                            G[i][h] = pp[1] * G[i+1][h - counter] + (1 - pp[1]) * (G[i+1][h] + 1)
                        else:
                            G[i][h] = (1 - pp[1]) * (G[i+1][h] + 1)
            
                # 最终期望伤害计算
                return E_dmg_no_counter * G[0][H] + E_dmg_counter * F[0][H]

            else:   
                assert self.special_type == 'magic'
                # 计算怪物对勇士每回合期望伤害
                monster_base = self._calculate_base_damage(self.m_atk, 0 , self.h_def_thresh)
                monster_dmg_per_turn = monster_base * (1 - u) * (1 + q)
                # 递推计算回合数期望（魔法师没有反弹和霸体的效果，但有破防这个效果）
                # 定义F[H]为怪物血量为H时，直到怪物死亡，勇士对怪物发起攻击的次数期望
                F = np.zeros(H+1)
                for h in range(1, H+1):
                    # F[H] = A * F[H] + B
                    A = 0
                    B = 1   
                    for i in range(5):
                        if mm[i]==0:
                            A += qq[i]
                        else:
                            B += qq[i] * F[max(0, h - mm[i])]
                    assert(A < 1 - 1e-6)
                    F[h] = B / (1 - A)
                expected_turns = F[H] - 1
                return monster_dmg_per_turn * expected_turns
            
        elif self.emblem_type == 'sage':
            hero_base = self._calculate_base_damage(self.h_atk, self.m_def, self.h_atk_thresh)
            if hero_base <= 0:
                return np.nan
                
            M = int(np.ceil(H / hero_base))
            effective_h_def = 0 if self.special_type == 'magic' else self.h_def
            monster_base = self._calculate_base_damage(self.m_atk, effective_h_def, self.h_def_thresh)
            
            if not self.special_type == 'magic' and monster_base==0 and self.h_atk > self.m_def:
                # 可能触发秒杀

                p1 = 1 - 0.16 * (1 - u)  # 每次怪物攻击不被秒杀的概率
                if self.emblem_level == 1:
                    monster_dmg_per_turn = - 0.61 * u * (self._round_half_up(self.m_atk / 5)) / p1
                elif self.emblem_level == 2:
                    monster_dmg_per_turn = - 0.81 * u * (self._round_half_up(self.m_atk / 5)) / p1
                else:
                    monster_dmg_per_turn = - u * self._round_half_up(self.m_atk / 5) / p1

                # F[M]为怪物剩余标准回合数为M时，直到怪物死亡，怪物对勇士进行的没有触发秒杀的攻击次数。
                F = np.zeros(M+1)
                for m in range(1, M+1):
                    # F[M] = A * F[M] + B
                    A = p * p1 ** K
                    B = p * (p1**K*(-p1)/(1-p1)+p1/(1-p1))
                    if m >= 2:
                        B += (1 - p) * (1 - v) * (p1**K * (F[m-1]-p1/(1-p1)) +p1/(1-p1) )
                    if m >= 3:
                        B += (1 - p) * v * (p1**K * (F[m-2]-p1/(1-p1)) +p1/(1-p1) )
                    assert(A < 1 - 1e-6)
                    F[m] = B / (1 - A)
                expected_turns = F[M]

                return monster_dmg_per_turn * expected_turns

            else:      
                # 不可能触发秒杀     
                monster_dmg_per_turn = monster_base * (1 - u) * (1 + q)
                
                sage_reduction_evasion_heal = 0
                sage_reduction_energy = 0
                if self.emblem_level == 1:
                    sage_reduction_evasion_heal = 0.61 * u * (self._round_half_up(self.m_atk / 5))
                elif self.emblem_level == 2:
                    sage_reduction_evasion_heal = 0.81 * u * (self._round_half_up(self.m_atk / 5))
                    if monster_base > 0 and self.special_type != 'magic':
                        sage_reduction_energy = (1 - u) * 0.11 * (50 * (1 + q) + 14.5)  # average heal 0~29 is 14.5
                else:
                    sage_reduction_evasion_heal = u * self._round_half_up(self.m_atk / 5)
                    if monster_base > 0 and self.special_type != 'magic':
                        sage_reduction_energy = (1 - u) * 0.31 * (50 * (1 + q) + 14.5)
                    elif self.special_type == 'magic':
                        sage_reduction_energy = (1 - u) * 0.21 * (100 * (1 + q) + self.m_atk + 9.5)  # average heal m_atk ~ m_atk+19 is m_atk+9.5

                monster_dmg_per_turn -= sage_reduction_evasion_heal + sage_reduction_energy
                monster_dmg_per_turn *= self.k_value
                expected_turns = 1 / ((1 - p) * ((1 + v) ** 2)) * (M * (1 + v) + v * (1 - (-v) ** M)) - 1
                
                return monster_dmg_per_turn * expected_turns

        elif self.emblem_type == 'hero':
            assert self.emblem_level >=2
            hero_base = self._calculate_base_damage(self.h_atk, self.m_def, self.h_atk_thresh)
            if hero_base <= 0:
                return np.nan

            effective_h_def = 0 if self.special_type == 'magic' else self.h_def
            monster_base = self._calculate_base_damage(self.m_atk, effective_h_def, self.h_def_thresh)
            monster_dmg_per_turn = self.k_value * monster_base * (1 - u) * (1 + q)

            # 勇士对怪物的进攻手段
            qq = np.zeros(5)
            mm = np.zeros(5, dtype=np.int64)
            if self.emblem_level == 3 and self.h_def < self.m_atk and self.special_type != 'solid':
                # 怪物闪避
                qq[0] = p
                mm[0] = 0
                # 勇士不暴击，无附加伤害
                qq[1] = (1-p) * (1-v) * (1-0.16)
                mm[1] = hero_base
                # 勇士不暴击，有附加伤害
                qq[2] = (1-p) * (1-v) * 0.16    
                mm[2] = hero_base + self._round_half_up(self.h_def / 4)
                # 勇士暴击, 无附加伤害
                qq[3] = (1-p) * v * (1-0.16)
                mm[3] = hero_base * 2
                # 勇士暴击, 有附加伤害
                qq[4] = (1-p) * v * 0.16
                mm[4] = (hero_base + self._round_half_up(self.h_def / 4)) * 2
            else:
                # 怪物闪避
                qq[0] = p
                mm[0] = 0
                # 勇士不暴击
                qq[1] = (1-p) * (1-v)
                mm[1] = hero_base
                # 勇士暴击
                qq[2] = (1-p) * v
                mm[2] = hero_base * 2
            
            # mm必须全部是非负整数
            if not all(isinstance(x, np.int64) and x >= 0 for x in mm):
                print(mm)
                raise Exception('Damage values mm must be non-negative integers in formula calculation.')
        
            # 计算回合数期望公式    
            # F[H]为怪物血量为H时，直到怪物死亡 ，勇士对怪物发起攻击的次数期望
            # F[H] = 1 + sum( qq[i] * F[H - mm[i]] )  (0<= i <=4)
            F = np.zeros(H+1)
            for h in range(1, H+1):
                # F[H] = A * F[H] + B
                A = 0
                B = 1   
                for i in range(5):
                    if mm[i]==0:
                        A += qq[i]
                    else:
                        B += qq[i] * F[max(0, h - mm[i])]
                assert(A < 1 - 1e-6)
                F[h] = B / (1 - A)
            
            # G[H]为怪物血量为H时，直到怪物死亡，勇士对怪物发起攻击的次数为奇数的概率
            # G[H] = 1 - sum( qq[i] * G[H - mm[i]] )  (0<= i <=4)
            G = np.zeros(H+1)
            for h in range(1, H+1):
                # G[H] = C * G[H] + D
                C = 0
                D = 1   
                for i in range(5):
                    if mm[i]==0:
                        C -= qq[i]
                    else:
                        D -= qq[i] * G[max(0, h - mm[i])]
                G[h] = D / (1 - C)

            expected_turns = (F[H] + G[H]) / 2 - 1
            return monster_dmg_per_turn * expected_turns

        else:
            raise Exception('Unknown emblem_type.')

    def calculate_abnormal_probability(self):
        """
        计算一次战斗后勇士进入异常状态的概率。
        """
        H = self.m_hp_max
        p, q = self.m_eva, self.m_crit
        u, v = self.h_eva, self.h_crit
        K = self.k_value

        if not self.have_abnormal:
            return 0.0
        
        can_instant_kill = False
        monster_base = self._calculate_base_damage(self.m_atk, self.h_def, self.h_def_thresh)
        if not self.special_type == 'magic' and monster_base==0 and self.h_atk > self.m_def:
            can_instant_kill = True

        if self.emblem_type is None or (self.emblem_type == 'hero' and self.emblem_level == 1) or self.emblem_type == 'sage':
            hero_base = self._calculate_base_damage(self.h_atk, self.m_def, self.h_atk_thresh)
            if hero_base <= 0:
                return np.nan
            M = int(np.ceil(H / hero_base))

            # 勇士对怪物的进攻手段
            qq = np.zeros(3)
            mm = np.zeros(3, dtype=np.int64)
            # 怪物闪避
            qq[0] = p
            mm[0] = 0
            # 普攻
            qq[1] = (1-p) * (1-v)
            mm[1] = 1
            # 暴击
            qq[2] = (1-p) * v
            mm[2] = 2

            # p1表示怪物每次攻击勇士后勇士进入异常状态的概率（对勇士闪避/秒杀的情况也进行了考虑）
            p1 = (1 - u) * self.abnormal_prob
            if can_instant_kill:
                p1 = (1 - 0.16) * (1 - u) * self.abnormal_prob

            # p2表示怪物每次攻击勇士后被勇士秒杀的概率（对勇士闪避的情况也进行了考虑）
            p2 = 0
            if can_instant_kill:
                p2 = 0.16 * (1 - u)

            # F[M]表示怪物剩余M次被勇士标准攻击后死亡，下一步为勇士攻击，勇士此时没有异常状态，而战后勇士进入异常状态的概率
            F = np.zeros(M+1)
            for m in range(1, M+1):
                # F[M] = A * F[M] + B
                A = 0
                B = 0
                for i in range(3):
                    if m - mm[i] > 0:
                        if mm[i]==0:
                            A += qq[i] * (1 - p1 - p2)**K
                            B += qq[i] * ((1-p1-p2)**K*(-p1/(p1+p2)) + p1/(p1+p2))
                        else:
                            B += qq[i] * ((1-p1-p2)**K*(F[m-mm[i]]-p1/(p1+p2)) + p1/(p1+p2))
                assert(A < 1 - 1e-6)
                F[m] = B / (1 - A)
            return F[M]
        elif self.emblem_type == 'hero':
            assert self.emblem_level >=2
            hero_base = self._calculate_base_damage(self.h_atk, self.m_def, self.h_atk_thresh)
            if hero_base <= 0:
                return np.nan

            # 勇士对怪物的进攻手段
            qq = np.zeros(5)
            mm = np.zeros(5, dtype=np.int64)
            if self.emblem_level == 3 and self.h_def < self.m_atk and self.special_type != 'solid':
                # 怪物闪避
                qq[0] = p
                mm[0] = 0
                # 勇士不暴击，无附加伤害
                qq[1] = (1-p) * (1-v) * (1-0.16)
                mm[1] = hero_base
                # 勇士不暴击，有附加伤害
                qq[2] = (1-p) * (1-v) * 0.16    
                mm[2] = hero_base + self._round_half_up(self.h_def / 4)
                # 勇士暴击, 无附加伤害
                qq[3] = (1-p) * v * (1-0.16)
                mm[3] = hero_base * 2
                # 勇士暴击, 有附加伤害
                qq[4] = (1-p) * v * 0.16
                mm[4] = (hero_base + self._round_half_up(self.h_def / 4)) * 2
            else:
                # 怪物闪避
                qq[0] = p
                mm[0] = 0
                # 勇士不暴击
                qq[1] = (1-p) * (1-v)
                mm[1] = hero_base
                # 勇士暴击
                qq[2] = (1-p) * v
                mm[2] = hero_base * 2

            # p1表示怪物每次攻击勇士后勇士进入异常状态的概率（对勇士闪避/秒杀的情况也进行了考虑）
            p1 = (1 - u) * self.abnormal_prob
            if can_instant_kill:
                p1 = (1 - 0.16) * (1 - u) * self.abnormal_prob

            # p2表示怪物每次攻击勇士后被勇士秒杀的概率（对勇士闪避的情况也进行了考虑）
            p2 = 0
            if can_instant_kill:
                p2 = 0.16 * (1 - u)

            # F[0][H]表示怪物生命值为H，下一步为勇士的第1次攻击，勇士此时没有异常状态，而战后勇士进入异常状态的概率
            # F[1][H]表示怪物生命值为H，下一步为勇士的第2次攻击，勇士此时没有异常状态，而战后勇士进入异常状态的概率
            # F[2][H]表示怪物生命值为H，下一步为怪物的大攻击，勇士此时没有异常状态，而战后勇士进入异常状态的概率
            # F[0][H] = sum( qq[i] * F[1][max(0,H - mm[i])]) )  (0<= i <=4)
            # F[1][H] = sum( qq[i] * F[2][max(0,H - mm[i])]) )  (0<= i <=4)
            # F[2][H] = (1-p1-p2)**K*(F[0][H]-p1/(p1+p2)) + p1/(p1+p2)
            F = np.zeros((3, H+1))
            for h in range(1, H+1):
                # 计算F[0][h]
                # 维护表达式：F[0][h] = A[i] * F[i][h] + B[i]    (1<= i <=2) 直到循环回到 F[0][h]
                A = np.zeros(3)
                B = np.zeros(3)
                for j in range(5):
                    if mm[j]==0:
                        A[1] += qq[j]
                    else:
                        B[1] += qq[j] * F[1][max(0, h - mm[j])]
                B[2] = B[1]
                for j in range(5):
                    if mm[j]==0:
                        A[2] += A[1] * qq[j]
                    else:
                        B[2] += A[1] * qq[j] * F[2][max(0, h - mm[j])]

                A[0] = A[2] * (1 - p1 - p2)**K
                B[0] = A[2] * ((1 - p1 - p2)**K * (-p1 / (p1 + p2)) + p1 / (p1 + p2)) + B[2]
                assert(A[0] < 1 - 1e-6)
                F[0][h] = B[0] / (1 - A[0])
                F[2][h] = (1 - p1 - p2)**K * (F[0][h] - p1 / (p1 + p2)) + p1 / (p1 + p2)
                for j in range(5):
                    F[1][h] += qq[j] * F[2][max(0, h - mm[j])]
            return F[0][H]

        elif self.emblem_type == 'overlord':
            lvl = self.emblem_level
            # 勇士对怪物的进攻手段
            qq = np.zeros(5)
            mm = np.zeros(5, dtype=np.int64)
            if lvl == 3:
                # 怪物闪避
                qq[0] = p
                mm[0] = 0
                # 普攻且不触发破防
                qq[1] = (1-p) * (1-v) * (1-0.11)
                mm[1] = self._calculate_base_damage(self.h_atk, self.m_def, self.h_atk_thresh) 
                # 普攻且触发破防
                qq[2] = (1-p) * (1-v) * 0.11
                reduction = self._round_half_up(self.m_def / 1.3)
                mm[2] = self._calculate_base_damage(self.h_atk, self.m_def - reduction, self.h_atk_thresh)
                # 暴击且不触发破防
                qq[3] = (1-p) * v * (1-0.11)
                mm[3] = self._calculate_base_damage(self.h_atk, self.m_def, self.h_atk_thresh) * 2
                # 暴击且触发破防
                qq[4] = (1-p) * v * 0.11
                mm[4] = self._calculate_base_damage(self.h_atk, self.m_def - reduction, self.h_atk_thresh) * 2
            else:
                # 怪物闪避
                qq[0] = p
                mm[0] = 0
                # 普攻
                qq[1] = (1-p) * (1-v)
                mm[1] = self._calculate_base_damage(self.h_atk, self.m_def, self.h_atk_thresh) 
                # 暴击
                qq[2] = (1-p) * v
                mm[2] = self._calculate_base_damage(self.h_atk, self.m_def, self.h_atk_thresh) * 2

            # mm必须全部是非负整数
            if not all(isinstance(x, np.int64) and x >= 0 for x in mm):
                print(mm)
                raise Exception('Damage values mm must be non-negative integers in formula calculation.')
            
            # Determine counter params per level
            if lvl == 1:
                counter_prob = 0.06
                overbody_prob = 0
                original_dmg_base = self._calculate_base_damage(self.m_atk, self.h_def, self.h_def_thresh)
                counter = self._round_half_up(original_dmg_base / 2)  # monster takes
                counter_big = counter
            elif lvl == 2:
                counter_prob = 0.11
                overbody_prob = 0.11
                original_dmg_base = self._calculate_base_damage(self.m_atk, self.h_def, self.h_def_thresh)
                counter = original_dmg_base
                counter_big = original_dmg_base
            else:
                counter_prob = 0.16
                overbody_prob = 0.16
                counter = self._round_half_up(self.m_atk / 3)
                counter_big = self._round_half_up(self.m_atk / 2)

            # 无法战斗判定
            if all(x == 0 for x in mm) and (counter == 0 or self.special_type == 'magic'):
                return np.nan                   

            p0 = self.abnormal_prob

            # p2表示怪物攻击勇士，没被闪避，没被反弹时，被勇士秒杀的概率
            p2 = 0
            if can_instant_kill:
                p2 = 0.16 * (1 - overbody_prob)  # 还不能有霸体

            # F[0][H]表示怪物血量为H，下一步为勇士攻击，勇士此时没有异常状态，而战后勇士进入异常状态的概率
            # F[i][H]表示怪物血量为H，下一步为怪物对勇士发起第i次攻击时，勇士此时没有异常状态，而战后勇士进入异常状态的概率
            # F[0][H] = sum( qq[i] * F[1][max(H - mm[i],0)] )  (0<= i <=4)
            # F[i][H] = u * F[i+1][H]      # 怪物攻击被闪避
            #         + (1 - u) * counter_prob * (p0 + (1-p0)*(F[i+1][H-counter]*indicator(H>counter_big)))   # 怪物攻击被反弹
            #         + (1 - u) * (1 - counter_prob) * (1 - p2) * (p0+(1-p0)*F[i+1][H]) # 怪物攻击未被闪避/反弹/秒杀
            #         # 怪物攻击被秒杀,此时概率为0
            F = np.zeros((K+2, H+1))
            for h in range(1, H+1):
                # 计算F[0][h]
                # 维护表达式：F[0][h] = A[i] * F[i][h] + B[i]    (1<= i <= K+1) 直到循环回到 F[0][h]
                # 为了方便起见，再维护一个表达式：F[i][h] = X[i] * F[i+1][h] + Y[i]   (1<= i <= K)
                A = np.zeros(K+2)
                B = np.zeros(K+2)
                X = np.zeros(K+2)
                Y = np.zeros(K+2)
                for j in range(5):
                    if mm[j]==0:
                        A[1] += qq[j]
                    else:
                        B[1] += qq[j] * F[1][max(0, h - mm[j])]
                for i in range(1, K+1):
                    X[i] = u + (1 - u) * (1 - counter_prob) * (1 - p2) * (1 - p0)
                    Y[i] = (1 - u) * counter_prob * p0 + (1 - u) * (1 - counter_prob) * (1 - p2) * p0
                    if h > counter_big:
                        if counter==0:
                            X[i] += (1 - u) * counter_prob * (1 - p0)
                        else:
                            Y[i] += (1 - u) * counter_prob * (1 - p0) * F[i+1][h - counter]
                    A[i+1] = X[i] * A[i]
                    B[i+1] = A[i] * Y[i] + B[i]
                assert(A[K+1] < 1 - 1e-6)
                F[0][h] = B[K+1] / (1 - A[K+1])
                F[K+1][h] = F[0][h]
                # 计算F[i][h] (1<= i <= K)
                for i in range(K, 0,-1):
                    F[i][h] = X[i] * F[i+1][h] + Y[i]
            return F[0][H]




def print_statistics(stats_dict):
    """
    辅助函数：漂亮地打印统计结果
    """
    print("-" * 30)
    print(f"【蒙特卡洛模拟统计结果】")
    print(f"模拟次数 (Count) : {stats_dict['count']}")
    
    if np.isnan(stats_dict['mean']):
        print("结果: 勇士无法击败怪物 (NaN)")
    else:
        print(f"平均值 (Mean)   : {stats_dict['mean']:.4f}")
        print(f"中位数 (Median) : {stats_dict['median']:.4f}")
        print(f"最大值 (Max)    : {stats_dict['max']:.4f}")
        print(f"最小值 (Min)    : {stats_dict['min']:.4f}")
        print(f"标准差 (Std Dev): {stats_dict['std']:.4f}")
    print("-" * 30)

def print_statistics_abnormal(stats_dict):
    """
    辅助函数：漂亮地打印异常状态统计结果
    """
    print("-" * 30)
    print(f"【蒙特卡洛模拟异常状态统计结果】")
    print(f"模拟次数     : {stats_dict['count']}")
    
    if np.isnan(stats_dict['abnormal_count']):
        print("结果: 勇士无法击败怪物 (NaN)")
    else:
        print(f"异常状态次数 : {stats_dict['abnormal_count']}")
        print(f"异常状态频率 : {stats_dict['abnormal_rate']:.4f}")
    print("-" * 30) 


def run_test_case(title, description, sim_params, n_trials=20000):
    """
    自动化测试运行器，包含假设检验。
    零假设 H0: 公式解析解 (formula_val) = 真实的数学期望
    """
    print("=" * 60)
    print(f"测试案例: {title}")
    print(f"描述: {description}")
    print("-" * 60)
    
    # 打印怪物属性 (省略，保持原样)
    print("【怪物属性】")
    print(f"  生命值(HP): {sim_params.get('m_hp', 'N/A')}")
    print(f"  攻击力(ATK): {sim_params.get('m_atk', 'N/A')}")
    print(f"  防御力(DEF): {sim_params.get('m_def', 'N/A')}")
    print(f"  闪避率(EVA): {sim_params.get('m_eva', 'N/A')}")
    print(f"  暴击率(CRIT): {sim_params.get('m_crit', 'N/A')}")
    
    # 打印勇士属性 (省略，保持原样)
    print("【勇士属性】")
    print(f"  攻击力(ATK): {sim_params.get('h_atk', 'N/A')}")
    print(f"  防御力(DEF): {sim_params.get('h_def', 'N/A')}")
    print(f"  闪避率(EVA): {sim_params.get('h_eva', 'N/A')}")
    print(f"  暴击率(CRIT): {sim_params.get('h_crit', 'N/A')}")
    
    # 打印其他参数 (省略，保持原样)
    h_atk_thresh = sim_params.get('h_atk_thresh', 0)
    h_def_thresh = sim_params.get('h_def_thresh', 0)
    print(f"【临界值】: 攻击 {h_atk_thresh}, 防御 {h_def_thresh}")

    special_type = sim_params.get('special_type',None)
    print(f"【特殊能力】: {special_type}")
    if special_type == 'k_combo':
        print(f"  连击次数: {sim_params.get('k_value', 1)}")
    
    emblem_type = sim_params.get('emblem_type', None)
    print(f"【徽章类型】: {emblem_type} (等级 {sim_params.get('emblem_level', 0)})")

    print("-" * 60)

    # 1. 初始化模拟器 (需要 MagicTowerSimulator 类可用)
    sim = MagicTowerSimulator(**sim_params)
    
    # 2. 计算解析解 (Formula)
    formula_val = sim.calculate_formula_expectation()
    
    # 3. 运行蒙特卡洛模拟 (Simulation)
    stats = sim.monte_carlo_simulation(n_trials)
    
    # 4. 打印结果对比
    
    # 针对 NaN/无法击败的情况
    if np.isnan(formula_val) or np.isnan(stats['mean']):
        print(f"【公式预期】: {'NaN' if np.isnan(formula_val) else f'{formula_val:.4f}'} (无法击败)")
        print(f"【模拟结果】: {'NaN' if np.isnan(stats['mean']) else f'{stats['mean']:.4f}'} (无法击败)")
        print_statistics(stats)
        return

    # 针对可击败的情况
    print(f"【公式预期】: {formula_val:.4f}")
    print(f"【模拟均值】: {stats['mean']:.4f}")
    
    # 计算误差
    error = abs(stats['mean'] - formula_val)
    error_pct = (error / np.abs(formula_val) * 100) if formula_val != 0 else np.nan
    print(f"【误差分析】: 绝对误差 {error:.4f} / 相对误差 {error_pct:.2f}%")
    
    # 5. 假设检验 (Z-Test)
    mean_sim = stats['mean']
    std_dev = stats['std']
    n = stats['count']
    mu_0 = formula_val

    # 避免除以零或标准差过小
    if std_dev == 0 or n < 30: # n < 30 可能用T检验，但此处沿用Z检验逻辑，并增加保护
        p_value = 1.0 if np.isclose(mean_sim, mu_0) else 0.0
    else:
        # Z 统计量计算: Z = (X̄ - μ₀) / (s / √n)
        z_stat = (mean_sim - mu_0) / (std_dev / np.sqrt(n))
        
        # P 值计算 (双尾检验): P = 2 * P(Z > |z_stat|)
        p_value = 2 * (1 - norm.cdf(np.abs(z_stat)))

    print(f"【假设检验P值】: {p_value:.6f} (显著性水平={SIGNIFICANCE_LEVEL})")
    
    # 6. 判定和异常抛出
    if p_value < SIGNIFICANCE_LEVEL:
        print("\n!!! 假设检验失败 !!!")
        print(f"P值 ({p_value:.6f}) < 显著性水平 ({SIGNIFICANCE_LEVEL})")
        print("=> 拒绝零假设：公式预期与模拟结果存在显著差异。")
        raise Exception(f"公式与模拟结果不符 (P值过低: {p_value:.6f})")
    else:
        print("假设检验通过: 公式预期与模拟结果无显著差异 (接受零假设)。")

    # 打印详细统计
    print_statistics(stats)
    print("=" * 60)

def run_test_case2(title, description, sim_params, n_trials=20000):
    """
    自动化测试运行器
    测试异常状态，并进行二项分布的 Z 假设检验。
    零假设 H0: 公式解析解 (formula_val) = 真实的异常状态概率
    """
    print("=" * 60)
    print(f"测试案例: {title}")
    print(f"描述: {description}")
    print("-" * 60)
    
    # 打印属性 (省略，保持原样)
    print("【怪物属性】")
    print(f"  生命值(HP): {sim_params.get('m_hp', 'N/A')}")
    print(f"  攻击力(ATK): {sim_params.get('m_atk', 'N/A')}")
    print(f"  防御力(DEF): {sim_params.get('m_def', 'N/A')}")
    print(f"  闪避率(EVA): {sim_params.get('m_eva', 'N/A')}")
    print(f"  暴击率(CRIT): {sim_params.get('m_crit', 'N/A')}")
    
    print("【勇士属性】")
    print(f"  攻击力(ATK): {sim_params.get('h_atk', 'N/A')}")
    print(f"  防御力(DEF): {sim_params.get('h_def', 'N/A')}")
    print(f"  闪避率(EVA): {sim_params.get('h_eva', 'N/A')}")
    print(f"  暴击率(CRIT): {sim_params.get('h_crit', 'N/A')}")
    
    h_atk_thresh = sim_params.get('h_atk_thresh', 0)
    h_def_thresh = sim_params.get('h_def_thresh', 0)
    print(f"【临界值】: 攻击 {h_atk_thresh}, 防御 {h_def_thresh}")

    special_type = sim_params.get('special_type',None)
    print(f"【特殊能力】: {special_type}")
    if special_type == 'k_combo':
        print(f"  连击次数: {sim_params.get('k_value', 1)}")
    
    emblem_type = sim_params.get('emblem_type', None)
    print(f"【徽章类型】: {emblem_type} (等级 {sim_params.get('emblem_level', 0)})")
    
    abnormal_prob_param = sim_params.get('abnormal_prob', 0.0)
    print(f"【异常状态参数】: {abnormal_prob_param:.4f}")
    
    print("-" * 60)
    
    # 1. 初始化模拟器
    sim = MagicTowerSimulator(**sim_params)
    
    # 2. 计算解析解 (Formula) - 理论概率 p0
    formula_val = sim.calculate_abnormal_probability()
    
    # 3. 运行蒙特卡洛模拟 (Simulation)
    stats = sim.monte_carlo_simulation_abnormal(n_trials)
    
    # 4. 打印结果对比
    
    # 针对 NaN/无法击败的情况 (假设NaN表示无法击败，无法计算概率)
    if np.isnan(formula_val) or np.isnan(stats['abnormal_rate']):
        print(f"【公式预期】: {'NaN' if np.isnan(formula_val) else f'{formula_val:.4f}'}")
        print(f"【模拟频率】: {'NaN' if np.isnan(stats['abnormal_rate']) else f'{stats['abnormal_rate']:.4f}'}")
        print("结果: 无法计算概率 (NaN)")
        print_statistics_abnormal(stats)
        return

    # 针对可计算概率的情况
    print(f"【公式预期】: {formula_val:.4f}")
    print(f"【模拟频率】: {stats['abnormal_rate']:.4f}")
    
    # 计算误差
    error = abs(stats['abnormal_rate'] - formula_val)
    print(f"【误差分析】: 绝对误差 {error:.4f}")
    
    # 5. 假设检验 (二项分布 Z-Test)
    
    p_hat = stats['abnormal_rate'] # 模拟频率
    p_0 = formula_val              # 理论概率 (零假设)
    n = n_trials                   # 样本量

    # 检查 Z 检验的条件 (np.isclose 避免浮点数比较问题)
    if np.isclose(p_0, 0.0) or np.isclose(p_0, 1.0) or n * p_0 < 5 or n * (1 - p_0) < 5:
        # 概率接近0或1，或样本量/概率太小，不适合用正态近似，直接比较
        # 此时 P值通常设置为 1.0 或 0.0
        p_value = 1.0 if np.isclose(p_hat, p_0, atol=2e-3) else 0.0
        print("注意: 概率接近0或1，或np太小，未进行严格Z检验。")
    else:
        # Z 统计量计算: Z = (p̂ - p₀) / SE
        standard_error = np.sqrt(p_0 * (1 - p_0) / n)
        z_stat = (p_hat - p_0) / standard_error
        
        # P 值计算 (双尾检验): P = 2 * P(Z > |z_stat|)
        p_value = 2 * (1 - norm.cdf(np.abs(z_stat)))

    print(f"【假设检验P值】: {p_value:.6f} (显著性水平={SIGNIFICANCE_LEVEL})")
    
    # 6. 判定和异常抛出
    if p_value < SIGNIFICANCE_LEVEL:
        print("\n!!! 假设检验失败 !!!")
        print(f"P值 ({p_value:.6f}) < 显著性水平 ({SIGNIFICANCE_LEVEL})")
        print("=> 拒绝零假设：公式预期与模拟频率存在显著差异。")
        raise Exception(f"公式与模拟结果不符 (P值过低: {p_value:.6f})")
    else:
        print("假设检验通过: 公式预期与模拟频率无显著差异 (接受零假设)。")

    # 打印详细统计
    print_statistics_abnormal(stats)
    print("=" * 60)

