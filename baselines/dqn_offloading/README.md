
<!-- dqn for task offloading baseline -->
# DQN for Task Offloading

This is a baseline implementation of the DQN algorithm for task offloading in a multi-agent system. The algorithm is implemented in the `dqn_offloading` directory.

Run the following command to train the DQN model:

```bash
cd airfogsim_code
python baselines/dqn_offloading/main_dqn_offloading.py
```

For testing the DQN model, modify `parseDQNArgs()` in `dqn_algorithm.py` to load the trained model:

```python
    parser.add_argument('--model_path', type=str, default='models/trans_dqn/model_34000.final_pth')
    parser.add_argument('--mode', type=str, default='test')
```

Then run the `main_dqn_offloading.py` script. This script will load the trained model and test it in the simulation environment.