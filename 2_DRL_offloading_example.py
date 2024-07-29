from airfogsim import AirFogSimEnv, AirFogSimScheduler, AirFogSimEnvVisualizer
import numpy as np
import yaml
import sys
import torch
import torch.nn.functional as F

def load_config(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)
    
class DQN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
# 1. Load the configuration file
config_path = sys.argv[1] if len(sys.argv) > 1 else 'example_1_config.yaml'
config = load_config(config_path)

# 2. Create the environment
env = AirFogSimEnv(config)

# 3. Create the environment visualizer (optional). 
env_wrapper = AirFogSimEnvVisualizer(env, config)

# 4. Get Schedulers
compSched = AirFogSimScheduler.getComputationScheduler()
compSched.setComputationModel(env, 'M/M/1')
taskSched = AirFogSimScheduler.getTaskScheduler() # task settings are in the config file
rewardSched = AirFogSimScheduler.getRewardScheduler()
rewardSched.setRewardModel(env, '1/delay')
entitySched = AirFogSimScheduler.getEntityScheduler()
agentSched = AirFogSimScheduler.getAgentScheduler()

# 5. Automatically initialize the DQN agent for each task node
input_dim = taskSched.getTaskFeatureDim(env) + 5 * entitySched.getFogNodeFeatureDim(env) + entitySched.getTaskNodeFeatureDim(env)
output_dim = 6 # 5 best fog nodes + local
sampleDQN = DQN(input_dim, 128, output_dim).to('cpu')
agentSched.initializeAgentByNodeType(env, node_type='TaskNode', agent=sampleDQN, agent_name='sampleDQN')


# 5. Start the simulation
env.reset()
done = False
while not done:
    # 5.1 Get all to-do tasks
    taskNodes = env.getTaskNodes()
    for task_node in taskNodes:
        tasks = task_node.getToDoTasks()
        agent = task_node.getAgentByName('sampleDQN')
        for task in tasks:
            # 5.2 Get the fog node with the highest computation power
            fog_node_list = entitySched.getNeighborFogNodesByNodeName(env, task_node['name'], sort_by='CPU', reverse=True)
            fog_node_list = fog_node_list[:5]
            # 5.3 Get the feature vector, in numpy
            fog_node_feature_list = [entitySched.getFogNodeFeatureByNodeName(env, fog_node['name']) for fog_node in fog_node_list]
            fog_node_feature_list = np.array(fog_node_feature_list).flatten()
            task_feature = taskSched.getTaskFeatureByTaskName(env, task['name'])
            task_node_feature = entitySched.getTaskNodeFeatureByNodeName(env, task_node['name'])
            feature = np.concatenate([task_feature, fog_node_feature_list, task_node_feature])
            feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)
            # 5.4 Get the action
            action = agent(feature)
            action = torch.argmax(action).item()
            selected_fog_node = fog_node_list[action]
            # 5.5 Offload the task to the fog node
            taskSched.offloadTaskToNode(env, task_node['name'], selected_fog_node['name'], task['name'])
    done = env.step()
    reward = 0
    for task_node in taskNodes:
        reward += rewardSched.getReward(env, task_node['name'])
    env_wrapper.render()
    print(f"Reward: {reward}")