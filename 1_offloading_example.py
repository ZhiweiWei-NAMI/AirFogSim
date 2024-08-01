from airfogsim import AirFogSimEnv, AirFogSimScheduler, AirFogSimEnvVisualizer
import numpy as np
import yaml
import sys

def load_config(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)
    

# 1. Load the configuration file
config_path = sys.argv[1] if len(sys.argv) > 1 else 'example_1_config.yaml'
config = load_config(config_path)

# 2. Create the environment
env = AirFogSimEnv(config)

# 3. Create the environment visualizer (optional). 
env_wrapper = AirFogSimEnvVisualizer(env, config)

# 4. Get Schedulers
compSched = AirFogSimScheduler.getComputationScheduler()
taskSched = AirFogSimScheduler.getTaskScheduler() # task settings are in the config file
taskSched.setTaskGenerationModel(env, 'Poisson', max_predictable_task_num=10) # max_predictable_task_num: always maintain 10 task info (generation time, cpu, ddl, priority, etc.) for future generation
rewardSched = AirFogSimScheduler.getRewardScheduler()
rewardSched.setRewardModel(env, '1/delay')
entitySched = AirFogSimScheduler.getEntityScheduler()
# 5. Start the simulation
env.reset()
done = False
while not done:
    # 5.1 Get all to-do tasks
    taskNodes = env.getTaskNodes()
    for task_node in taskNodes:
        tasks = task_node.getToDoTasks()
        for task in tasks:
            # 5.2 Get the fog node with the highest computation power
            fog_node_list = entitySched.getNeighborFogNodesByNodeName(env, task_node['name'], sort_by='CPU', reverse=True)
            # returned fog_node_list is a list of dict, which can be easily sorted by lambda function
            # 5.3 Offload the task to the fog node
            taskSched.offloadTaskToNode(env, task_node['name'], fog_node_list[0]['name'], task['name'])
    done = env.step()
    reward = 0
    for task_node in taskNodes:
        reward += rewardSched.getReward(env, task_node['name'])
    env_wrapper.render()
    print(f"Reward: {reward}")
env_wrapper.close()