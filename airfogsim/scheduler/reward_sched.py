from sympy import symbols, log, sympify
from sympy.core.sympify import SympifyError
from .base_sched import BaseScheduler
from ..airfogsim_env import AirFogSimEnv

class RewardScheduler(BaseScheduler):
    """The reward scheduler for the reinforcement learning agents. Provide static methods to compute the reward for the reinforcement learning agents. Use sympy to parse the reward expression and compute the reward for each task node. According to test, the efficiency of evalf is similar to direct symbolic computation.
    """
    REWARD_MODEL = None
    SYMOBOLS = {}
    ACCEPTED_SYMBOLS = ['task_delay', 'energy', 'task_ratio']

    @staticmethod
    def setRewardModel(env:AirFogSimEnv, expression):
        """Set the reward model for the reinforcement learning agents. Note that reward is computed for each task node, according to the QoE metrics defined in the expression. The QoE metrics include task delay, energy consumption, and task ratio. This metric is updated each step for each task node.
        
        Args:
            env (AirFogSimEnv): The environment.
            expression (str): The reward expression. The expression should be a valid sympy expression

        Raises:
            ValueError: If the expression is invalid.

        Examples:
            >>> sched = RewardScheduler()
            >>> sched.setRewardModel('1/log(1+energy)')
        """
        try:
            # 使用sympify来将字符串表达式转换为可计算的表达式
            RewardScheduler.REWARD_MODEL = sympify(expression)
            # 自动检测表达式中使用的变量，并创建符号
            RewardScheduler.SYMOBOLS = {str(sym): symbols(str(sym)) for sym in RewardScheduler.REWARD_MODEL.free_symbols}
            # 检查是否所有符号都是有效的
            for sym in RewardScheduler.SYMOBOLS:
                if sym not in RewardScheduler.ACCEPTED_SYMBOLS:
                    raise ValueError(f"Invalid symbol in reward expression: {sym}, expected one of {RewardScheduler.ACCEPTED_SYMBOLS}")
        except SympifyError as e:
            raise ValueError(f"Invalid expression: {e}")

    @staticmethod
    def getRewardByTaskNodeName(env:AirFogSimEnv, task_node_name:str):
        """Compute the reward for the reinforcement learning agents. Here, the reward is computed for each task node only.

        Args:
            env (AirFogSimEnv): The environment.
        """
        # 从env中获取任务节点的信息
        task_node = env.getTaskNodeByName(task_node_name)
        # 从RewardScheduler.SYMOBOLS中获取任务节点的信息，用task_node.getXXX()来获取
        kwargs = {key: getattr(task_node, key) for key in RewardScheduler.SYMOBOLS}

        # 获取任务信息，需要是还没有
        if not all(param in kwargs for param in RewardScheduler.SYMOBOLS):
            missing = list(set(RewardScheduler.SYMOBOLS) - set(kwargs))
            raise ValueError(f"Missing parameters for reward computation: {missing}")

        # 替换表达式中的符号为实际的参数值
        subs = {RewardScheduler.SYMOBOLS[key]: kwargs[key] for key in RewardScheduler.SYMOBOLS}
        return RewardScheduler.REWARD_MODEL.evalf(subs=subs)
    
