from airfogsim import AirFogSimEnv, AirFogSimScheduler, AirFogSimEnvVisualizer
import numpy as np
import yaml
import sys

def load_config(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)
    

# 1. Load the configuration file
config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.yaml'
config = load_config(config_path)

# 2. Create the environment
env = AirFogSimEnv(config)

# 3. Get Schedulers and setup the environment
compSched = AirFogSimScheduler.getComputationScheduler()
taskSched = AirFogSimScheduler.getTaskScheduler()
taskSched.setTaskGenerationModel(env, 'Poisson')
taskSched.setTaskNodePossibility(env, node_types=['vehicle'], max_num=30, threshold_poss=0.5)
rewardSched = AirFogSimScheduler.getRewardScheduler()
rewardSched.setRewardModel(env, '1/delay')
entitySched = AirFogSimScheduler.getEntityScheduler()
commSched = AirFogSimScheduler.getCommunicationScheduler()


while not env.isDone():
    # 5.1 Get all to-do tasks
    all_task_infos = taskSched.getAllToOffloadTaskInfos(env)
    all_node_infos = entitySched.getAllNodeInfos(env)
    # convert all_node_infos to dict, where key is the node id
    all_node_infos_dict = {}
    for node_info in all_node_infos:
        all_node_infos_dict[node_info['id']] = node_info
    for task_dict in all_task_infos:
        task_node_id = task_dict['task_node_id']
        task_id = task_dict['task_id']
        task_node = all_node_infos_dict[task_node_id]
        neighbor_infos = entitySched.getNeighborNodeInfosById(env, task_node_id, sorted_by='distance')
        nearest_node_id = neighbor_infos[0]['id']
        taskSched.setTaskOffloading(env, task_node_id, task_id, nearest_node_id)
    all_offloading_task_infos = taskSched.getAllOffloadingTaskInfos(env)
    n_RB = commSched.getNumberOfRB(env)
    for task_dict in all_offloading_task_infos:
        allocated_RB_nos = np.random.choice(n_RB, 3, replace=False)
        commSched.setCommunicationWithRB(env, task_dict['task_id'], allocated_RB_nos)
    env.step()
    reward = 0
    last_step_succ_task_infos = taskSched.getLastStepSuccTaskInfos(env)
    for task_info in last_step_succ_task_infos:
        reward += rewardSched.getRewardByTask(env, task_info)
    print(f"Reward: {reward}")
env.close()