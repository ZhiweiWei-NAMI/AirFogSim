from ..airfogsim_env import AirFogSimEnv
from .base_sched import BaseScheduler
class EntityScheduler(BaseScheduler):
    """The entity scheduler for entities.
    """
    
    @staticmethod
    def setMaxVehicleNumber(env: AirFogSimEnv, max_vehicle_number: int):
        """Set the maximum vehicle number.

        Args:
            env (AirFogSimEnv): The environment.
            max_vehicle_number (int): The maximum vehicle number.

        Returns:
            bool: The flag to indicate whether the maximum vehicle number is set successfully.
        """
        return True
    
    @staticmethod
    def setVehicleDisappearAfterArrival(env: AirFogSimEnv, disappear_after_arrival: bool):
        """Set the flag to indicate whether the vehicle disappears after arrival.

        Args:
            env (AirFogSimEnv): The environment.
            disappear_after_arrival (bool): The flag to indicate whether the vehicle disappears after arrival.

        Returns:
            bool: The flag to indicate whether the flag is set successfully.
        """
        return True

    @staticmethod
    def setVehicleTrafficModel(env: AirFogSimEnv, traffic_model: str):
        """Set the vehicle traffic model.

        Args:
            env (AirFogSimEnv): The environment.
            traffic_model (str): The traffic model.

        Returns:
            bool: The flag to indicate whether the traffic model is set successfully.
        """
        return True
    
    @staticmethod
    def setVehicleArrivalRate(env: AirFogSimEnv, arrival_rate: float):
        """Set the vehicle arrival rate.

        Args:
            env (AirFogSimEnv): The environment.
            arrival_rate (float): The vehicle arrival rate.

        Returns:
            bool: The flag to indicate whether the vehicle arrival rate is set successfully.
        """
        return True
    
    @staticmethod
    def getNeighborVehiclesByNodeId(env: AirFogSimEnv, node_id: str, distance: float):
        """Get the neighbor vehicles by the node id.

        Args:
            env (AirFogSimEnv): The environment.
            node_id (str): The node id.
            distance (float): The distance.

        Returns:
            list: The neighbor vehicles in dict format.
        """
        return []
    
    @staticmethod
    def getNeighborRSUsByNodeId(env: AirFogSimEnv, node_id: str, distance: float):
        """Get the neighbor RSUs by the node id.

        Args:
            env (AirFogSimEnv): The environment.
            node_id (str): The node id.
            distance (float): The distance.

        Returns:
            list: The neighbor RSUs in dict format.
        """
        return []
    
    @staticmethod
    def getNeighborUAVsByNodeId(env: AirFogSimEnv, node_id: str, distance: float):
        """Get the neighbor UAVs by the node id.

        Args:
            env (AirFogSimEnv): The environment.
            node_id (str): The node id.
            distance (float): The distance.

        Returns:
            list: The neighbor UAVs in dict format.
        """
        return []
    
    @staticmethod
    def getVehicleIds(env: AirFogSimEnv):
        """Get the vehicle ids.

        Args:
            env (AirFogSimEnv): The environment.

        Returns:
            list: The vehicle ids.
        """
        return []
    
    @staticmethod
    def getUAVIds(env: AirFogSimEnv):
        """Get the UAV ids.

        Args:
            env (AirFogSimEnv): The environment.

        Returns:
            list: The UAV ids.
        """
        return []
    
    @staticmethod
    def getRSUIds(env: AirFogSimEnv):
        """Get the RSU ids.

        Args:
            env (AirFogSimEnv): The environment.

        Returns:
            list: The RSU ids.
        """
        return []
    
    @staticmethod
    def getCloudServerIds(env: AirFogSimEnv):
        """Get the cloud server ids.

        Args:
            env (AirFogSimEnv): The environment.

        Returns:
            list: The cloud server ids.
        """
        return []
    
    @staticmethod
    def getRegionIds(env: AirFogSimEnv):
        """Get the region ids.

        Args:
            env (AirFogSimEnv): The environment.

        Returns:
            list: The region ids.
        """
        return []
    
    @staticmethod
    def getV2VChannelIds(env: AirFogSimEnv):
        """Get the V2V channel ids.

        Args:
            env (AirFogSimEnv): The environment.

        Returns:
            list: The V2V channel ids.
        """
        return []
    
    @staticmethod
    def getV2UChannelIds(env: AirFogSimEnv, reverse = False):
        """Get the V2U channel ids.

        Args:
            env (AirFogSimEnv): The environment.
            reverse (bool): The flag to indicate whether the V2U channel ids or U2V channel ids are returned.

        Returns:
            list: The V2U or U2V channel ids.
        """
        return []
    
    @staticmethod
    def getV2RChannelIds(env: AirFogSimEnv, reverse = False):
        """Get the V2R channel ids.

        Args:
            env (AirFogSimEnv): The environment.
            reverse (bool): The flag to indicate whether the V2R channel ids or R2V channel ids are returned.

        Returns:
            list: The V2R or R2V channel ids.
        """
        return []
    
    @staticmethod
    def getU2UChannelIds(env: AirFogSimEnv):
        """Get the U2U channel ids.

        Args:
            env (AirFogSimEnv): The environment.

        Returns:
            list: The U2U channel ids.
        """
        return []
    
    @staticmethod
    def getU2RChannelIds(env: AirFogSimEnv, reverse = False):
        """Get the U2R channel ids.

        Args:
            env (AirFogSimEnv): The environment.
            reverse (bool): The flag to indicate whether the U2R channel ids or R2U channel ids are returned.

        Returns:
            list: The U2R or R2U channel ids.
        """
        return []
    
    @staticmethod
    def getR2RChannelIds(env: AirFogSimEnv):
        """Get the R2R channel ids.

        Args:
            env (AirFogSimEnv): The environment.

        Returns:
            list: The R2R channel ids.
        """
        return []
    
