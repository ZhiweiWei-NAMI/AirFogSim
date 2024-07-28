from .entities.fog_node import FogNode
class AirFogSimEnv():
    """AirFogSimEnv is the main class for the airfogsim environment. It provides the simulation of communication, computation, storage, battery, vehicle/UAV trajectory, cloud/cloudlet nodes, AI models for entities, blockchain, authentication, and privacy. It also provides the APIs for the agent to interact with the environment. The agent can be a DRL agent, a rule-based agent, or a human player.
    """

    def __init__(self, config):
        """The constructor of the AirFogSimEnv class. It initializes the environment with the given configuration.

        Args:
            config (dict): The configuration of the environment. Please follow standard YAML format.
        """
        pass

    def step(self):
        """The step function of the environment. It simulates the environment for one time step.

        Returns:
            bool: The done signal. True if the episode is done, False otherwise.
        """
        return True
    
    def getFogVById(self, id):
        """Get the fog node by the given id.

        Args:
            id (int): The id of the fog node.

        Returns:
            FogNode: The fog node.
        """
        return FogNode(0, "fogV", (0, 0), 100, 100)
    
    def getNumberOfChannelsByType(self, type):
        """Get the number of channels by the given type.

        Args:
            type (str): The type of the channel.

        Returns:
            int: The number of channels.
        """
        return 1

