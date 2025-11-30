
from .base_sched import BaseScheduler

class TaskScheduler(BaseScheduler):
    """Scheduler for task scheduling, setting the task generation model and executing the task offloading.
    """

    @staticmethod
    def setTaskGenerationModel(env, model, predictable_seconds = 1, **kwargs):
        """Set the task generation model for the environment. The task generation model will not affact the determined task infos, so better to set it before the simulation starts.

        Args:
            env (AirFogSimEnv): The AirFogSim environment.
            model (str): The task generation model, including 'Poisson', 'Random', etc.
            predictable_seconds (float, optional): The maximum predictable seconds for the task generation. Defaults to 1, which means tasks will be generated during the first second.

        Examples:
            taskSched.setTaskGeneration(env, 'Poisson', predictable_seconds=1)
        """
        env.task_manager.setTaskGenerationModel(model, **kwargs)
        env.task_manager.setPredictableSeconds(predictable_seconds)

    @staticmethod
    def getAllToOffloadTaskInfos(env, check_dependency=False):
        """Get the task infos for the environment to offload.

        Args:
            env (AirFogSimEnv): The AirFogSim environment.

        Returns:
            list: The list of task infos.

        Examples:
            taskSched.getAllToOffloadTaskInfos(env)
        """
        task_dict = env.task_manager.getWaitingToOffloadTasks()
        task_info_list = []
        for task_node_id, tasks in task_dict.items():
            for task in tasks:
                flag = True
                if check_dependency:
                    flag = env.task_manager.checkTaskDependency(task_node_id, task.getTaskId())
                if flag == True:
                    task_info_list.append(task.to_dict())
        return task_info_list
    
    @staticmethod
    def getTaskInfoByTaskNodeAndTaskId(env, task_node_id, task_id):
        """Get the task info by task node id and task id.

        Args:
            env (AirFogSimEnv): The AirFogSim environment.
            task_node_id (str): The task node id.
            task_id (str): The task id.

        Returns:
            dict: The task info.

        Examples:
            taskSched.getTaskInfoByTaskNodeAndTaskId(env, 'UAV_1', 'Task_1')
        """
        task = env.task_manager.getTaskByTaskNodeAndTaskId(task_node_id, task_id)
        return task.to_dict()
    
    @staticmethod
    def moveComputingTaskToOffloading(env, old_compute_node_id, task_node_id, task_id, new_compute_node_id):
        """Move the task to waiting to offload.

        Args:
            env (AirFogSimEnv): The AirFogSim environment.
            old_compute_node_id (str): The old compute node id.
            task_node_id (str): The task node id.
            task_id (str): The task id.
            new_compute_node_id (str): The new compute node id.

        Examples:
            taskSched.moveTaskToWaitingToOffload(env, 'Fog_1', 'UAV_1', 'Task_1', 'Fog_2')
        """
        task = None
        task_list = env.task_manager.getToComputeTasks(old_compute_node_id)
        for t in task_list:
            if t.getTaskId() == task_id:
                task = t
                break
        task_list.remove(task)
        env.task_manager.setToComputeTasks(old_compute_node_id, task_list)
        if task is not None:
            to_offload_tasks = env.task_manager.getOffloadingTasksByNodeId(task_node_id)
            # 把task的target_node_id设置为new_compute_node_id
            task.changeOffloadTo(new_compute_node_id, [new_compute_node_id], env.simulation_time)
            to_offload_tasks.append(task)
            env.task_manager.setOffloadingTasksByNodeId(task_node_id, to_offload_tasks)
    
    @staticmethod
    def getToOffloadTaskNumberByTaskNode(env, task_node_id):
        """Get the number of the tasks for the task node.

        Args:
            env (AirFogSimEnv): The AirFogSim environment.
            task_node_id (str): The task node id.

        Returns:
            int: The number of the tasks.

        Examples:
            taskSched.getTaskNumberByTaskNode(env, 'UAV_1')
        """
        return len(env.task_manager.getToOffloadTasks(task_node_id))

    @staticmethod
    def getAllToOffloadTasks(env, check_dependency=False):
        """Get the tasks for the environment to offload.

        Args:
            env (AirFogSimEnv): The AirFogSim environment.

        Returns:
            list: The list of task infos.

        Examples:
            taskSched.getAllToOffloadTaskInfos(env)
        """
        task_dict = env.task_manager.getWaitingToOffloadTasks()
        task_info_list = []
        for task_node_id, tasks in task_dict.items():
            for task in tasks:
                flag = True
                if check_dependency:
                    flag = env.task_manager.checkTaskDependency(task_node_id, task.getTaskId())
                if flag:
                    task_info_list.append(task)
        return task_info_list
    
    @staticmethod
    def getAllTaskDAGs(env):
        """Get the task DAGs for the environment.

        Args:
            env (AirFogSimEnv): The AirFogSim environment.

        Returns:
            dict: The task DAGs. The key is the task node id, and the value is the task DAG (nx.DiGraph) for the task ID.

        Examples:
            taskSched.getAllTaskDAGs(env)
        """
        task_dags = env.task_manager._task_dependencies
        # 以 task_node为key，value为None或nx.DiGraph()
        return task_dags

    @staticmethod
    def getAllOffloadingTaskInfos(env):
        """Get the task infos for the environment to offload.

        Args:
            env (AirFogSimEnv): The AirFogSim environment.

        Returns:
            list: The list of task infos.

        Examples:
            taskSched.getAllOffloadingTaskInfos(env)
        """
        task_dict = env.task_manager.getOffloadingTasks()
        task_info_list = []
        for task_node_id, tasks in task_dict.items():
            for task in tasks:
                task_info_list.append(task.to_dict())
        return task_info_list

    @staticmethod
    def setTaskOffloading(env, task_node_id:str, task_id:str, target_node_id:str, route:list=None):
        """Set the task offloading for the task node.

        Args:
            env (AirFogSimEnv): The AirFogSim environment.
            task_node_id (str): The task node id.
            task_id (str): The task id.
            target_node_id (str): The target node id.
            route (list): The route for the offloading task.

        Returns:
            bool: The result of the offloading task.
        Examples:
            taskSched.setTaskOffloading(env,'UAV_1','Task_1','UAV_3')
        """
        return env.task_manager.offloadTask(task_node_id, task_id, target_node_id, env.simulation_time, route)
    
    @staticmethod
    def setTaskAttribute(env, task_node_id:str, task_id:str, attribute:str, value):
        """Set the attribute of the task.

        Args:
            env (AirFogSimEnv): The AirFogSim environment.
            task_node_id (str): The task node id.
            task_id (str): The task id.
            attribute (str): The attribute name.
            value: The value of the attribute.

        Examples:
            taskSched.setTaskAttribute(env,'UAV_1','Task_1','task_deadline',10)
        """
        task = env.task_manager.getTaskByTaskNodeAndTaskId(task_node_id, task_id)
        task.setAttribute(attribute, value)
    
    @staticmethod
    def getAllComputingTaskInfos(env):
        """Get the task infos for the environment to offload.

        Args:
            env (AirFogSimEnv): The AirFogSim environment.

        Returns:
            list: The list of task infos.

        Examples:
            taskSched.getAllComputingTaskInfos(env)
        """
        task_dict = env.task_manager.getComputingTasks()
        task_info_list = []
        for task_node_id, tasks in task_dict.items():
            for task in tasks:
                task_info_list.append(task.to_dict())
        return task_info_list
    
    @staticmethod
    def getLastStepSuccTaskInfos(env):
        """Get the success task infos for last timeslot.

        Args:
            env (AirFogSimEnv): The AirFogSim environment.

        Returns:
            list: The list of task infos.

        Examples:
            taskSched.getLastStepSuccTaskInfos(env)
        """
        recently_done_100_tasks = env.task_manager.getRecentlyDoneTasks()
        last_step = env.simulation_time - env.traffic_interval
        task_info_list = []
        for task in recently_done_100_tasks:
            if task.isFinished() and task.getLastOperationTime() >= last_step:
                task_info_list.append(task.to_dict())
        return task_info_list
    
    @staticmethod
    def getLastStepFailTaskInfos(env):
        """Get the failed task infos for last timeslot.

        Args:
            env (AirFogSimEnv): The AirFogSim environment.

        Returns:
            list: The list of task infos.

        Examples:
            taskSched.getLastStepFailTaskInfos(env)
        """
        recently_fail_100_tasks = env.task_manager.getRecentlyFailedTasks()
        last_step = env.simulation_time - env.traffic_interval
        task_info_list = []
        for task in recently_fail_100_tasks:
            if task.isFinished() and task.getLastOperationTime() >= last_step:
                task_info_list.append(task.to_dict())
        return task_info_list

    @staticmethod
    def generateTaskOfMission(env,mission_task_profile):
        """Generate task by profile.

        Args:
            env (AirFogSimEnv): The AirFogSim environment.
            mission_task_profile (dict): {task_node_id,task_deadline,arrival_time}

        Returns:
            Task: Generated task.

        Examples:
            taskSched.generateTaskOfMission(env,mission_task_profile)
        """
        task_node_id=mission_task_profile['task_node_id']
        task_deadline=mission_task_profile['task_deadline']
        arrival_time=mission_task_profile['arrival_time']
        return_size=mission_task_profile['return_size']
        task=env.task_manager.generateTaskInfoOfMission(task_node_id,task_deadline,arrival_time,return_size, return_lazy_set=True)
        return task

    @staticmethod
    def getWaitingToReturnTaskInfos(env):
        """Get waiting to return task infos.

         Args:

         Returns:
             dict: node_id -> {task:Task,...}

         Examples:
             taskSched.getWaitingToReturnTaskInfos(env)
         """
        task_infos=env.task_manager.getWaitingToReturnTaskInfos()
        return task_infos

    @staticmethod
    def setTaskReturnRoute(env,task_id,return_route):
        env.task_return_routes[task_id]=return_route

    @staticmethod
    def getDoneTaskNum(env):
        """Get the number of the success tasks.

        Args:
            env (AirFogSimEnv): The AirFogSim environment.

        Returns:
            int: The number of the success tasks.

        Examples:
            taskSched.getSuccessTaskNum(env)
        """
        return len(env.task_manager.getDoneTasks())
    
    @staticmethod
    def getDoneTaskNumLessThanSeconds(env, seconds):
        """Get the number of the tasks finished in the last seconds.

        Args:
            env (AirFogSimEnv): The AirFogSim environment.
            seconds (float): The seconds.

        Returns:
            int: The number of the tasks finished in the last seconds.

        Examples:
            taskSched.getTaskNumLessThanSeconds(env, 10)
        """
        done_task_num = TaskScheduler.getDoneTaskNum(env)
        out_of_ddl_tasks = env.task_manager.getOutOfDDLTasks()
        for task in out_of_ddl_tasks:
            if task.task_delay < seconds:
                done_task_num += 1
        return done_task_num

    @staticmethod
    def getLastStepDoneTaskDelay(env):
        """Get the delay of the success tasks for last timeslot.

        Args:
            env (AirFogSimEnv): The AirFogSim environment.

        Returns:
            list: The list of task delays.

        Examples:
            taskSched.getLastStepDoneTaskDelay(env)
        """
        recently_done_100_tasks = env.task_manager.getRecentlyDoneTasks()
        last_step = env.simulation_time - env.traffic_interval
        task_delay_list = []
        for task in recently_done_100_tasks:
            if task.isFinished() and task.getLastOperationTime() >= last_step:
                task_delay_list.append(task.task_delay)
        return task_delay_list

    @staticmethod
    def getTotalTaskNum(env):
        """Get the number of the total tasks.

        Args:
            env (AirFogSimEnv): The AirFogSim environment.

        Returns:
            int: The number of the total tasks.

        Examples:
            taskSched.getTotalTaskNum(env)
        """
        return len(env.task_manager.getAllTasks())
    
    @staticmethod
    def getOutOfDDLTasks(env):
        """Get the number of the failed tasks.

        Args:
            env (AirFogSimEnv): The AirFogSim environment.

        Returns:
            int: The number of the failed tasks.

        Examples:
            taskSched.getFailedTaskNum(env)
        """
        return len(env.task_manager.getOutOfDDLTasks())