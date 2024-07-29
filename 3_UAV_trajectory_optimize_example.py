
from airfogsim import AirFogSimEnv, AirFogSimScheduler, AirFogSimEnvVisualizer
import numpy as np
import yaml
import sys

def load_config(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)
    

# 1. Load the configuration file
config_path = sys.argv[1] if len(sys.argv) > 1 else 'example_3_config.yaml'
config = load_config(config_path)

# 2. Create the environment
env = AirFogSimEnv(config)

# 3. Create the environment visualizer (optional).
env_wrapper = AirFogSimEnvVisualizer(env, config)

# 4. Get Schedulers
entitySched = AirFogSimScheduler.getEntityScheduler()
print(entitySched.getUAVFeatureDim(env))
print(entitySched.getUAVFeatureNames(env)) # UAV features are set in class variable of entityScheduler
commSched = AirFogSimScheduler.getCommunicationScheduler()

# 5. Start the simulation
env.reset()
done = False
while not done:
    # 5.1 Get all UAV Info
    uav_names = env.getUAVNames()
    uav_features = []
    for uav_name in uav_names:
        uav_features.append(entitySched.getUAVFeatureByNodeName(env, uav_name))
    # print(uav_features)
    # 5.2 Get the region with the lowest SINR
    regions = entitySched.getRegionNames(env) # Regions are set corresponding to RSU locations
    sinr = []
    for region in regions:
        sinr.append(commSched.getRegionSINR(env, region))
    # print(sinr)
    # 5.3 Move the UAV to the region with the lowest SINR
    sorted_regions = [x for _, x in sorted(zip(sinr, regions))]
    for i, uav_name in enumerate(uav_names):
        entitySched.moveUAVToRegion(env, uav_name, sorted_regions[i%len(sorted_regions)])
    done = env.step()
    env_wrapper.render()
env_wrapper.close()
    