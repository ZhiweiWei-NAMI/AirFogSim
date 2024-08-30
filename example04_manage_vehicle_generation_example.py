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
entitySched = AirFogSimScheduler.getEntityScheduler()
topoSched = AirFogSimScheduler.getTopologyScheduler()
topoSched.setGridWidth(env, 10) # 10m as the grid width
# RSU is the region center
topoSched.setRegionPartition(env, 'Voronoi') # Voronoi partition
topoSched.setRegionPartition(env, 'Radius', 100) # Radius partition with 100m as the radius 
# if choose Radius partition, there might be some overlap between regions, and nodes in the overlap region will be assigned to the region with the smallest index. The accessed RSU can be changed by the commScheduler.accessRSUByNodeName(env, node_name)

# 5. Start the simulation
env.reset()
done = False
while not done:
    # 5.1 Get all Lane Info
    lane_names = entitySched.getLaneNames(env)
    # 5.2 Generate vehicle traffic from Lane 1 to Lane 2
    route = topoSched.getPossibleRoutesByLaneNames(env, 'Lane1', 'Lane2') # generate vehicle traffic from Lane 1 to Lane 2, traci type
    entitySched.addVehicleByRoute(env, route)
    done = env.step()
    env_wrapper.render()