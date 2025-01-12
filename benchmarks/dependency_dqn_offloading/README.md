
<!-- dqn for task offloading benchmark -->
# Transformer-based DQN for Dependency-aware Task Offloading in Multi-agent Systems

This is an implementation of the DQN algorithm for dependency-aware task offloading in multi-agent systems. For task dependencies (DAG), a pre-trained GAE (Graph Attention Encoder) is used to encode the task graph for each agent. Then, the DQN algorithm is based on the transformer model, which is used to learn the dependency between tasks and offloading decisions. 
The DQN algorithm is implemented in the `dep_dqn_algorithm.py` file.

Run the following command to train the DQN model:

```bash
cd airfogsim_code
python benchmarks/dependency_dqn_offloading/main_dep_dqn_offloading.py
```

For testing the DQN model, modify `parseDQNArgs()` in `dep_dqn_algorithm.py` to load the trained model:

```python
    parser.add_argument('--model_path', type=str, default='models/dep_trans_dqn/final_model.final_pth')
    parser.add_argument('--mode', type=str, default='test')
```
For task dependencies, a directed acyclic graph (DAG) is used to represent the dependencies between tasks. The probability of an edge between two tasks in the DAG can be set to control the dependency between tasks. To change the probability of an edge, modify `dag_edge_prob` in `dep_dqn_algorithm_config.yaml`:

```yaml
task_profile: # task profile for each node type
  task_node_gen_poss: 0.8 # The possibility of generating a task node when inializing the node
  task_node_profiles: [{'type':'UAV', 'max_node_num': 10}, {'type':'vehicle', 'max_node_num': 40}] # The types of task nodes and the maximum number of nodes of each type
  vehicle: # The profile of vehicles
    lambda: 2
    dag_edge_prob: 0.2 # The probability of generating an edge in the DAG of task graph
  uav: # The profile of UAVs
    lambda: 2
    dag_edge_prob: 0.2 # The probability of generating an edge in the DAG of task graph
```

Then run the `main_dep_dqn_offloading.py` script. This script will load the trained model and test it in the simulation environment.
