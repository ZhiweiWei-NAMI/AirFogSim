from .scheduler import *
import inspect
class AirFogSimScheduler:
    """The scheduler for AirFogSimEnv. It provides communication, computation, task generation, blockchain scheduler. Agent interacts with this class to get the state, reward, and done signal.
    """
    @staticmethod
    def getComputationScheduler():
        """Get the computation scheduler for the environment. It schedules the computation tasks for the fog nodes.

        Returns: 
            ComputationScheduler: The computation scheduler.
        """
        return ComputationScheduler()
    
    @staticmethod
    def getCommunicationScheduler():
        """Get the communication scheduler for the environment. It schedules the communication tasks for the fog nodes.

        Returns: 
            CommunicationScheduler: The communication scheduler.
        """
        return CommunicationScheduler()
    
    @staticmethod
    def getTaskScheduler():
        """Get the task scheduler for the environment. It schedules the tasks for the task nodes.

        Returns: 
            TaskScheduler: The task scheduler.
        """
        return TaskScheduler()
    
    @staticmethod
    def getBlockchainScheduler():
        """Get the blockchain scheduler for the environment. It schedules the blockchain tasks for the blockchain nodes.

        Returns: 
            BlockchainScheduler: The blockchain scheduler.
        """
        return BlockchainScheduler()
    
    @staticmethod
    def getRewardScheduler():
        """Get the reward scheduler for the environment. It calculates the reward for the agent.

        Returns: 
            RewardScheduler: The reward scheduler.
        """
        return RewardScheduler()
    
    @staticmethod
    def getSchedulerMethodList():
        """Get the list of all scheduler methods. Use inspect module to get the list of all static methods in this class. The method name should start with 'get' and end with 'Scheduler'.

        Returns: 
            list: The list of all scheduler method names.
        """
        method_list = []
        for name, method in inspect.getmembers(AirFogSimScheduler):
            if inspect.ismethod(method) and name.startswith('get') and name.endswith('Scheduler'):
                method_list.append(name)
        return method_list