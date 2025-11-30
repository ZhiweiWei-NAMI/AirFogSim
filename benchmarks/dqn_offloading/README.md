
<!-- dqn for task offloading benchmark -->
# Simple DQN for Task Offloading

This is an implementation of the DQN algorithm for task offloading in a multi-agent system. The algorithm is implemented in the `dqn_offloading` directory.

Run the following command to train the DQN model:

```bash
cd airfogsim_code
python benchmarks/dqn_offloading/main_dqn_offloading.py
```

For testing the DQN model, modify `parseDQNArgs()` in `dqn_algorithm.py` to load the trained model:

```python
    parser.add_argument('--model_path', type=str, default='models/trans_dqn/final_model.final_pth')
    parser.add_argument('--mode', type=str, default='test')
```

Then run the `main_dqn_offloading.py` script. This script will load the trained model and test it in the simulation environment.

Different reward functions or hyperparameters can be set in `dqn_airfogsim_config.yaml`.