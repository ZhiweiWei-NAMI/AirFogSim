from sympy import symbols, log, sympify
from sympy.core.sympify import SympifyError

class RewardScheduler:
    def __init__(self):
        self.reward_model = None
        self.symbols = {}
        self.accepted_symbols = ['delay', 'energy', 'cpu', 'deadline', 'size', 'priority']

    def setRewardModel(self, expression):
        try:
            # 使用sympify来将字符串表达式转换为可计算的表达式
            self.reward_model = sympify(expression)
            # 自动检测表达式中使用的变量，并创建符号
            self.symbols = {str(sym): symbols(str(sym)) for sym in self.reward_model.free_symbols}
            # 检查是否所有符号都是有效的
            for sym in self.symbols:
                if sym not in self.accepted_symbols:
                    raise ValueError(f"Invalid symbol in reward expression: {sym}, expected one of {self.accepted_symbols}")
        except SympifyError as e:
            raise ValueError(f"Invalid expression: {e}")

    def computeReward(self, **kwargs):
        # 检查是否所有必要的参数都已提供
        if not all(param in kwargs for param in self.symbols):
            missing = list(set(self.symbols) - set(kwargs))
            raise ValueError(f"Missing parameters for reward computation: {missing}")

        # 替换表达式中的符号为实际的参数值
        subs = {self.symbols[key]: value for key, value in kwargs.items()}
        return self.reward_model.evalf(subs=subs)
    

if __name__ == "__main__":
    sched = RewardScheduler()
    sched.setRewardModel('1/log(1+energy)')
    key_value = {'delay': 1}
    reward = sched.computeReward(**key_value)
    print(reward)