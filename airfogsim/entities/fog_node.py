class FogNode:
    """The fog node class. This class represents the basic fog node entity in the airfogsim environment.
    """

    def __init__(self, fog_node_id:int, fog_node_type:str, fog_node_location:tuple, fog_node_capacity:float, fog_node_resource:float):
        """Initialize the fog node.

        Args:
            fog_node_id (int): The fog node ID.
            fog_node_type (str): The fog node type, e.g., "cloud", "fog", "edge".
            fog_node_location (tuple): The fog node location, e.g., (x, y).
            fog_node_capacity (float): The fog node capacity.
            fog_node_resource (float): The fog node resource.
        """
        self.fog_node_id = fog_node_id
        self.fog_node_type = fog_node_type
        self.fog_node_location = fog_node_location
        self.fog_node_capacity = fog_node_capacity
        self.fog_node_resource = fog_node_resource