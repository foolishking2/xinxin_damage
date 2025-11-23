import random
import numpy as np

class MagicTowerSimulator:

    def _round_half_up(self, value):
            """
            Helper function to implement standard mathematical rounding (round half up).
            执行四舍五入到整数。
            """
            return int(value + 0.5 + 1e-8)

    def __init__(self, 
                 m_hp, m_atk, m_def, m_eva, m_crit,  # m=怪物的，hp=生命值，atk=攻击力，def=防御力，eva=闪避率,crit=暴击率
                 h_atk, h_def, h_eva, h_crit,        # h=勇士的
                 h_atk_thresh=0, h_def_thresh=0,     # 攻击临界值，防御临界值
                 special_type=None,  # 怪物的特殊能力，只允许 'solid', 'mimic', 'magic', 'k_combo' ，None
                 k_value=1,          # 怪物的连击次数，仅当 special_type='k_combo' 时有效
                 emblem_type=None,   # 章的类型 (None, 'sage', 'hero', 'overlord')
                 emblem_level=0):    # 章的等级 (0-3)
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
            "h_eva": h_eva, "h_crit": h_crit
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

        # 战前属性调整 (坚固/仿攻)
        self.m_def = self.m_def_raw
        if self.special_type == 'solid':
            if self.h_atk > self.m_def_raw:
                self.m_def = self.h_atk
                
        self.m_atk = self.m_atk_raw
        if self.special_type == 'mimic':
            if self.h_atk > self.m_atk_raw:
                self.m_atk = self.h_atk

    # 这个函数必须返回非负整数
    def _calculate_base_damage(self, atk, defense, threshold):
        assert isinstance(atk, int) and isinstance(defense, int) and isinstance(threshold, int)
        if atk > defense:
            return atk - defense
        elif defense - threshold <= atk <= defense:
            return 1
        else:
            return 0

    def simulate_once(self):
        """
        模拟一次完整的战斗过程，返回勇士受到的总伤害。
        """
        
        is_ol_3 = self.emblem_type == 'overlord' and self.emblem_level == 3
        is_non_magic = self.special_type != 'magic'

        # ==================================================
        # NEW UNWINNABLE CHECK: 确保至少有一种方式能对怪物造成伤害
        # ==================================================
        
        # 1. 勇士标准攻击伤害潜力
        max_potential_hero_dmg = self._calculate_base_damage(self.h_atk, self.m_def, self.h_atk_thresh)
        
        # 2. 勇士破防 (Breach) 攻击伤害潜力
        if is_ol_3:
            # 计算最大的防御减少量（即破防成功）
            reduction_max = self._round_half_up(self.m_def / 1.3)
            m_def_min = max(0, self.m_def - reduction_max)
            
            # 计算破防成功时的伤害
            breach_dmg = self._calculate_base_damage(self.h_atk, m_def_min, self.h_atk_thresh)
            
            # 取标准攻击伤害和破防伤害中的最大值
            max_potential_hero_dmg = max(max_potential_hero_dmg, breach_dmg)
            
        # 3. 反弹 (Counter-Attack) 伤害潜力
        counter_dmg_potential = 0
        if is_ol_3 and is_non_magic:
            # 怪物承受伤害 = 怪物攻击力 / 3
            counter_dmg_potential = self._round_half_up(self.m_atk / 3)

        # 最终判定：如果所有伤害来源都不能对怪物造成伤害，则战斗无法胜利 (np.nan)
        if max_potential_hero_dmg <= 0 and counter_dmg_potential <= 0:
             return np.nan
        
        # ==================================================
        # END NEW UNWINNABLE CHECK
        # ==================================================
        
        current_m_hp = self.m_hp_max
        total_hero_damage_taken = 0
        
        # 预设怪物攻击阶段的初始防御力（处理魔攻）
        h_def = 0 if self.special_type == 'magic' else self.h_def
        
        while True:
            # =========================
            # 1. 勇士发起攻击
            # =========================
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
                
                # D. 判定勇士暴击
                if random.random() < self.h_crit:
                    damage_to_monster = current_hero_base_dmg * 2
                else:
                    damage_to_monster = current_hero_base_dmg
            
            current_m_hp -= damage_to_monster
            
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
                else:
                    # --- 命中：检查霸者3级效果 (非魔法师系怪物) ---
                    
                    if is_non_magic and is_ol_3:
                        
                        # 2.1 16% 反弹 (Counter) - 独立判定，第一优先级
                        if random.random() < 0.16: 
                            # 1. 判定瞬杀
                            atk_half_rounded = self._round_half_up(self.m_atk / 2)
                            if atk_half_rounded >= current_m_hp:
                                current_m_hp = 0
                                break # 怪物被瞬杀，跳出 K-combo 循环
                            
                            # 2. 伤害结算
                            original_dmg_base = self._calculate_base_damage(self.m_atk, h_def, self.h_def_thresh)
                            
                            hero_takes = self._round_half_up(original_dmg_base / 3)
                            monster_takes = self._round_half_up(self.m_atk / 3)
                            
                            total_hero_damage_taken += hero_takes
                            current_m_hp -= monster_takes
                            
                            if current_m_hp <= 0:
                                assert(0) # 理论上不可能发生，因为瞬杀已经处理
                            continue # 反弹成功，跳过本次攻击的后续判定和伤害计算
                        
                        # 2.2 16% 霸体 (Overlord Body) - 独立判定，仅在反弹未触发时执行
                        if random.random() < 0.16:
                            current_h_def_for_calc *= 2 # 防御力翻倍，用于接下来的伤害计算
                    
                    # --- 标准伤害计算 ---
                    
                    # 1. 计算基础伤害
                    monster_base_dmg = self._calculate_base_damage(self.m_atk, current_h_def_for_calc, self.h_def_thresh)
                    
                    # 2. 判定怪物暴击
                    if random.random() < self.m_crit:
                        damage_to_hero = monster_base_dmg * 2
                    else:
                        damage_to_hero = monster_base_dmg
                    
                    total_hero_damage_taken += damage_to_hero
                
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

    def calculate_formula_expectation(self):
        """
        计算期望伤害精确值的公式解。
        """
        H = self.m_hp_max
        p, q = self.m_eva, self.m_crit
        u, v = self.h_eva, self.h_crit
        K = self.k_value
        
        if self.emblem_type is None:
            hero_base = self._calculate_base_damage(self.h_atk, self.m_def, self.h_atk_thresh)
            if hero_base <= 0:
                return np.nan
                
            M = np.ceil(H / hero_base)
            effective_h_def = 0 if self.special_type == 'magic' else self.h_def
            monster_base = self._calculate_base_damage(self.m_atk, effective_h_def, self.h_def_thresh)
            monster_dmg_per_turn = self.k_value * monster_base * (1 - u) * (1 + q)
            
            expected_turns = 1 / ((1 - p) * ((1 + v) ** 2)) * (M * (1 + v) + v * (1 - (-v) ** M)) - 1
            
            return monster_dmg_per_turn * expected_turns
        elif self.emblem_type == 'overlord' and self.emblem_level == 3:

            # 勇士对怪物的进攻手段
            qq = np.zeros(5)
            mm = np.zeros(5, dtype=np.int64)
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
            # mm必须全部是非负整数
            if not all(isinstance(x, np.int64) and x >= 0 for x in mm):
                print(mm)
                raise Exception('Damage values mm must be non-negative integers in formula calculation.')
            # 反弹伤害
            counter = self._round_half_up(self.m_atk / 3)
            counter_big = self._round_half_up(self.m_atk / 2)
            # 无法战斗判定
            if all(x == 0 for x in mm) and counter == 0:
                return np.nan   

            if self.special_type != 'magic':           
                # 怪物对勇士的进攻手段
                pp = np.zeros(6)
                nn = np.zeros(6)
                # 触发闪避
                pp[0] = u
                nn[0] = 0
                # 触发反弹
                pp[1] = (1-u) * 0.16
                original_dmg_base = self._calculate_base_damage(self.m_atk, self.h_def, self.h_def_thresh)
                nn[1] = self._round_half_up(original_dmg_base / 3)            
                # 普通攻击不触发霸体
                pp[2] = (1-u) * (1-q) * (1-0.16) * (1-0.16)
                nn[2] = self._calculate_base_damage(self.m_atk, self.h_def, self.h_def_thresh)
                # 普通攻击触发霸体
                pp[3] = (1-u) * (1-q) * (1-0.16) * 0.16
                nn[3] = self._calculate_base_damage(self.m_atk, self.h_def * 2, self.h_def_thresh)
                # 暴击攻击不触发霸体
                pp[4] = (1-u) * q * (1-0.16) * (1-0.16)
                nn[4] = self._calculate_base_damage(self.m_atk, self.h_def, self.h_def_thresh) * 2
                # 暴击攻击触发霸体
                pp[5] = (1-u) * q * (1-0.16) * 0.16
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
                    # 维护表达式：F[0][h] = A[i] * F[i][h] + B[i]    (1<= i <= K) 直到循环回到 F[0][h]
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
                    assert(A[K+1] < 1)
                    F[0][h] = B[K+1] / (1 - A[K+1])
                    F[K+1][h] = F[0][h]
                    # 计算F[i][h] (1<= i <= K)
                    for i in range(K, 0,-1):
                        if h > counter_big:
                            F[i][h] = pp[1] * (F[i+1][h - counter] + 1) + (1 - pp[1]) * F[i+1][h]
                        else:
                            F[i][h] = (1 - pp[1]) * F[i+1][h]
                    # 计算G[0][h]
                    # 维护表达式：G[0][h] = C[i] * G[i][h] + D[i]    (1<= i <= K) 直到循环回到 G[0][h]
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
                    assert(C[K+1] < 1)
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
                monster_dmg_per_turn = self.k_value * monster_base * (1 - u) * (1 + q)
                # 递推计算回合数期望（魔法师没有反弹这个效果，但有破防这个效果）
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
                    assert(A < 1)
                    F[h] = B / (1 - A)
                expected_turns = F[H] - 1
                return monster_dmg_per_turn * expected_turns


        else:
            raise Exception('Other emblem condition is not supported yet in formula calculation.')

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


def run_test_case(title, description, sim_params, n_trials=20000):
    """
    自动化测试运行器
    :param title: 测试标题
    :param description: 测试目的描述
    :param sim_params: 传递给模拟器的参数字典
    :param n_trials: 模拟次数
    """
    print("=" * 60)
    print(f"测试案例: {title}")
    print(f"描述: {description}")
    print("-" * 60)
    
    # 打印怪物属性
    print("【怪物属性】")
    print(f"  生命值(HP): {sim_params.get('m_hp', 'N/A')}")
    print(f"  攻击力(ATK): {sim_params.get('m_atk', 'N/A')}")
    print(f"  防御力(DEF): {sim_params.get('m_def', 'N/A')}")
    print(f"  闪避率(EVA): {sim_params.get('m_eva', 'N/A')}")
    print(f"  暴击率(CRIT): {sim_params.get('m_crit', 'N/A')}")
    
    # 打印勇士属性
    print("【勇士属性】")
    print(f"  攻击力(ATK): {sim_params.get('h_atk', 'N/A')}")
    print(f"  防御力(DEF): {sim_params.get('h_def', 'N/A')}")
    print(f"  闪避率(EVA): {sim_params.get('h_eva', 'N/A')}")
    print(f"  暴击率(CRIT): {sim_params.get('h_crit', 'N/A')}")
    
    # 打印其他参数
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

    # 1. 初始化模拟器
    sim = MagicTowerSimulator(**sim_params)
    
    # 2. 计算解析解 (Formula)
    formula_val = sim.calculate_formula_expectation()
    
    # 3. 运行蒙特卡洛模拟 (Simulation)
    stats = sim.monte_carlo_simulation(n_trials)
    
    # 4. 打印结果对比
    if np.isnan(formula_val) and np.isnan(stats['mean']):
        print(f"【公式预期】: NaN (无法击败)")
        print(f"【模拟结果】: NaN (无法击败)")
    else:
        print(f"【公式预期】: {formula_val:.4f}")
        print(f"【模拟均值】: {stats['mean']:.4f}")
        
        # 计算误差
        error = abs(stats['mean'] - formula_val)
        error_pct = (error / formula_val * 100) if formula_val != 0 else np.nan
        print(f"【误差分析】: 绝对误差 {error:.4f} / 相对误差 {error_pct:.2f}%")
        
        # 打印详细统计
        print_statistics(stats)
