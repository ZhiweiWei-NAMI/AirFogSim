from sympy import symbols, log, sympify
from sympy.core.sympify import SympifyError
from .base_sched import BaseScheduler

class RewardScheduler(BaseScheduler):
    """The reward scheduler for the reinforcement learning agents. Provide static methods to compute the reward for the reinforcement learning agents. Use sympy to parse the reward expression and compute the reward for each task node. According to test, the efficiency of evalf is similar to direct symbolic computation.
    """
    REWARD_MODEL = None
    SYMOBOLS = None
    ACCEPTED_SYMBOLS = ['energy', 'task_ratio', 'task_deadline', 'task_delay',
                        '_mission_duration_sum','_mission_arrival_time','_mission_start_time','_mission_deadline','_mission_finish_time'] # mission_duration is an array

    @staticmethod
    def setRewardModel(env, expression):
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
            RewardScheduler.REWARD_MODEL = {env: sympify(expression)} if RewardScheduler.REWARD_MODEL is None else {**RewardScheduler.REWARD_MODEL, env: sympify(expression)}
            # 自动检测表达式中使用的变量，并创建符号
            RewardScheduler.SYMOBOLS = {env: {str(sym): symbols(str(sym)) for sym in RewardScheduler.REWARD_MODEL[env].free_symbols}} if RewardScheduler.SYMOBOLS is None else {**RewardScheduler.SYMOBOLS, env: {str(sym): symbols(str(sym)) for sym in RewardScheduler.REWARD_MODEL[env].free_symbols}}
            # 检查是否所有符号都是有效的
            for sym in RewardScheduler.SYMOBOLS[env]:
                if sym not in RewardScheduler.ACCEPTED_SYMBOLS:
                    raise ValueError(f"Invalid symbol in reward expression: {sym}, expected one of {RewardScheduler.ACCEPTED_SYMBOLS}")
        except SympifyError as e:
            raise ValueError(f"Invalid expression: {e}")

    @staticmethod
    def getRewardByTask(env, task_info:dict):
        """Compute the reward of the task.

        Args:
            env (AirFogSimEnv): The environment.
            task_info (dict): The task information.
        """
        if RewardScheduler.REWARD_MODEL is None:
            raise ValueError("Reward model is not set, please set the reward model first.")
        if RewardScheduler.SYMOBOLS is None:
            raise ValueError("Symbols are not set, please set the reward model first.")
        if env not in RewardScheduler.REWARD_MODEL or env not in RewardScheduler.SYMOBOLS:
            raise ValueError(f"Reward model is not set for {env}, please set the reward model first.")
        task = env.task_manager.getDoneTaskByTaskNodeAndTaskId(task_info['task_node_id'], task_info['task_id'])
        # 调用task.getXXX()获取任务信息
        kwargs = {key: task.__getattribute__(key) for key in RewardScheduler.SYMOBOLS[env]}

        # 获取任务信息，需要是还没有
        if not all(param in kwargs for param in RewardScheduler.SYMOBOLS[env]):
            missing = list(set(RewardScheduler.SYMOBOLS[env]) - set(kwargs))
            raise ValueError(f"Missing parameters for reward computation: {missing}")

        # 替换表达式中的符号为实际的参数值
        subs = {RewardScheduler.SYMOBOLS[env][key]: float(kwargs[key]) for key in RewardScheduler.SYMOBOLS[env]}
        return RewardScheduler.REWARD_MODEL[env].evalf(subs=subs)

    @staticmethod
    def getRewardByMission(env, mission_info: dict):
        """Compute the reward of the task.

        Args:
            env (AirFogSimEnv): The environment.
            mission_info (dict): The mission information (the dict is output of the 'to_dict' function in mission object).
        """
        if RewardScheduler.REWARD_MODEL is None:
            raise ValueError("Reward model is not set, please set the reward model first.")
        if RewardScheduler.SYMOBOLS is None:
            raise ValueError("Symbols are not set, please set the reward model first.")
        if env not in RewardScheduler.REWARD_MODEL or env not in RewardScheduler.SYMOBOLS:
            raise ValueError(f"Reward model is not set for {env}, please set the reward model first.")
        mission = env.mission_manager.getDoneMissionByMissionNodeAndMissionId(mission_info['appointed_node_id'], mission_info['mission_id'])
        # 调用task.getXXX()获取任务信息
        kwargs = {key: mission.__getattribute__(key) for key in RewardScheduler.SYMOBOLS[env]}

        # 获取任务信息，需要是还没有
        if not all(param in kwargs for param in RewardScheduler.SYMOBOLS[env]):
            missing = list(set(RewardScheduler.SYMOBOLS[env]) - set(kwargs))
            raise ValueError(f"Missing parameters for reward computation: {missing}")

        # 替换表达式中的符号为实际的参数值
        subs = {RewardScheduler.SYMOBOLS[env][key]: float(kwargs[key]) for key in RewardScheduler.SYMOBOLS[env]}
        return RewardScheduler.REWARD_MODEL[env].evalf(subs=subs)
