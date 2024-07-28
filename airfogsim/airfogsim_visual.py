from .airfogsim_env import AirFogSimEnv

class AirFogSimEnvVisualizer():
    def __init__(self, env:AirFogSimEnv, config):
        """The visualizer for AirFogSimEnv. It provides the visualization of the environment. Agent interacts with this class to get the visualized state, reward, and done signal.
        
        Args:
            env (AirFogSimEnv): The AirFogSimEnv environment.
            config (dict): The configuration of the environment. Please follow standard YAML format.
        """
        self.env = env
        self.config = config