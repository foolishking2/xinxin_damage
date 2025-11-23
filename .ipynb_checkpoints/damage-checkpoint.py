import random

class BattleSimulator:
    def __init__(self, 
                 m_hp, m_atk, m_def, m_eva, m_crit, 
                 h_atk, h_def, h_eva, h_crit):
        """
        初始化战斗双方属性
        """
        # 怪物属性 (Monster)
        self.m_hp_max = m_hp
        self.m_atk = m_atk
        self.m_def = m_def
        self.m_eva = m_eva   # 怪物闪避率
        self.m_crit = m_crit # 怪物暴击率
        
        # 勇士属性 (Hero)
        self.h_atk = h_atk
        self.h_def = h_def
        self.h_eva = h_eva   # 勇士闪避率
        self.h_crit = h_crit # 勇士暴击率

    def simulate_once(self):
        """
        模拟一次完整的战斗，逻辑显式展开在循环内。
        """
        current_m_hp = self.m_hp_max
        total_hero_damage_taken = 0
        
        while True:
            # =========================
            # 1. 勇士发起攻击 (Hero Turn)
            # =========================
            
            # A. 判定怪物闪避 (优先级最高)
            # 产生一个 0-1 的随机数，如果小于怪物闪避率，则攻击无效
            if random.random() < self.m_eva:
                damage_to_monster = 0
            else:
                # B. 未闪避，计算基础伤害
                base_dmg = self.h_atk - self.m_def
                
                # C. 判定勇士暴击
                # 注意：这里是条件概率，前提是未闪避
                if random.random() < self.h_crit:
                    damage_to_monster = base_dmg * 2
                else:
                    damage_to_monster = base_dmg
            
            # D. 结算伤害
            current_m_hp -= damage_to_monster
            
            # E. 死亡判定 (勇士攻击后立刻检查)
            if current_m_hp <= 0:
                break
            
            # =========================
            # 2. 怪物发起反击 (Monster Turn)
            # =========================
            
            # A. 判定勇士闪避
            if random.random() < self.h_eva:
                damage_to_hero = 0
            else:
                # B. 未闪避，计算基础伤害
                base_dmg = self.m_atk - self.h_def
                
                # C. 判定怪物暴击
                if random.random() < self.m_crit:
                    damage_to_hero = base_dmg * 2
                else:
                    damage_to_hero = base_dmg
            
            # D. 结算伤害 (累计到勇士受到的总伤害中)
            total_hero_damage_taken += damage_to_hero
            
        return total_hero_damage_taken

    def monte_carlo_simulation(self, n_trials=10000):
        """
        执行蒙特卡洛模拟
        """
        total_damage_acc = 0
        for _ in range(n_trials):
            total_damage_acc += self.simulate_once()
            
        return total_damage_acc / n_trials

# --- 测试代码 ---
if __name__ == "__main__":
    # 示例参数
    battle = BattleSimulator(
        m_hp=100, m_atk=20, m_def=5, m_eva=0.3, m_crit=0.2,
        h_atk=25, h_def=10, h_eva=0.2, h_crit=0.5
    )

    # 运行模拟
    avg_dmg = battle.monte_carlo_simulation(100000)
    print(f"勇士受到伤害的期望值: {avg_dmg:.2f}")