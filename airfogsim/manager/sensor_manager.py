import random
from ..entities.sensor import Sensor

from traffic_manager import TrafficManager


class SensorManager:
    """ Sensor Manager is responsible for generating and deploying sensors and managing the sensor status.
    """
    NODE_TYPE = ['vehicle', 'UAV', 'RSU']
    STATE=['idle','busy','all']

    def __init__(self, config_sensing, traffic_manager):
        assert set(config_sensing['node_type']).issubset(
            SensorManager.NODE_TYPE), 'The node type is not supported. Only support {}'.format(SensorManager.NODE_TYPE)
        self._sensors = {} # key: node_id, value: list of sensors
        self._busy_sensors = {} # key: node_id, value: list of busy sensors
        self._idle_sensors = {} # key: node_id, value: list of idle sensors
        self._unavailable_sensors = {} # key: node_id, value: list of unavailable sensors
        self._sensor_id_counter = 0
        self._sensors_per_node = config_sensing['sensors_per_node']
        self._node_type = config_sensing['node_type']
        self._sensor_type_num = config_sensing['sensor_type_num']
        self._initializeSensors(traffic_manager)

    def __getNewSensorId(self):
        self._sensor_id_counter += 1
        return self._sensor_id_counter

    def _initializeSensors(self, traffic_manager: TrafficManager):
        UAV_numbers = traffic_manager.getNumberOfUAVs()
        UAV_infos = traffic_manager.getUAVTrafficInfos()
        vehicle_numbers = traffic_manager.getNumberOfVehicles()
        vehicle_infos = traffic_manager.getVehicleTrafficInfos()

        self._generateSensors(UAV_infos)
        self._generateSensors(vehicle_infos)

    def _generateSensors(self, node_infos):
        for node_id, _ in node_infos:
            self._sensors[node_id]=[]
            for idx in range(self._sensors_per_node):
                new_sensor_type = 'Sensor_type_' + str(random.randint(1, self._sensor_type_num))
                new_sensor_accuracy = random.random()  # 随机生成0-1之间的精度
                new_sensor_id = 'Sensor_' + str(self.__getNewSensorId())
                new_sensor = Sensor(new_sensor_id, new_sensor_type, new_sensor_accuracy, node_id)
                self._sensors[node_id].append(new_sensor)
                self._idle_sensors[node_id].append(new_sensor)

    def _getSensor(self,sensors_dict, sensor_id):
        target_sensor=None
        target_node_id=None
        for node_id, sensors in sensors_dict:
            for sensor in sensors:
                if sensor.getSensorId()==sensor_id:
                    target_sensor= sensor
                    target_node_id=node_id
        return target_node_id,target_sensor

    def _addSensor(self,sensors_dict, node_id,sensor):
        if node_id not in sensors_dict:
            sensors_dict[node_id]=[]
            sensors_dict[node_id].append(sensor)

    def _removeSensor(self,sensors_dict, node_id,sensor):
        if node_id not in sensors_dict:
            return
        else:
            sensors_dict[node_id].remove(sensor)


    def startUseById(self, sensor_id):
        """start use the sensor.

        Args:
            sensor_id (str): The sensor id.

        Returns:

        Examples:
            sensor_manager.startUseById('Sensor_1')
        """
        node_id,sensor=self._getSensor(self._sensors,sensor_id)
        assert sensor != None
        sensor.startUse()
        self._removeSensor(self._idle_sensors,node_id,sensor_id)
        self._addSensor(self._busy_sensors,node_id,sensor_id)

    def endUseById(self, sensor_id):
        """stop use the sensor.

        Args:
            sensor_id (str): The sensor id.

        Returns:

        Examples:
            sensor_manager.stopUseById('Sensor_1')
        """
        node_id,sensor = self._getSensor(self._sensors, sensor_id)
        assert sensor != None
        sensor.endUse()
        self._removeSensor(self._busy_sensors,node_id,sensor_id)
        self._addSensor(self._idle_sensors,node_id, sensor_id)

    def getUsableById(self, sensor_id):
        """check if the sensor is usable.

        Args:
            sensor_id (str): The sensor id.

        Returns:

        Examples:
            sensor_manager.getUsableById('Sensor_1')
        """
        _,sensor = self._getSensor(self._sensors, sensor_id)
        assert sensor != None
        return sensor.isUsable()

    def setAvailavleById(self, sensor_id, available):
        """set available state of sensor.

        Args:
            sensor_id (str): The sensor id.
            available (bool): If the sensor is available

        Returns:

        Examples:
            sensor_manager.getUsableById('Sensor_1')
        """
        _,sensor = self._getSensor(self._sensors, sensor_id)
        assert sensor != None
        sensor.setAvailable(available)

    def getSensorsByType(self,state,type):
        assert state in SensorManager.STATE,'The state is not supported. Only support {}'.format(SensorManager.STATE)
        target_sensors_dict={}
        if state=='idle':
            sensors_dict=self._idle_sensors
        elif state=='busy':
            sensors_dict = self._busy_sensors
        elif state=='all':
            sensors_dict = self._sensors

        for node_id,sensors in sensors_dict:
            target_sensors_dict[node_id]=[]
            for sensor in sensors:
                if sensor.getSensorType()==type:
                    target_sensors_dict[node_id].append(sensor)
            if len(target_sensors_dict[node_id])==0:
                target_sensors_dict.pop(node_id)
        return target_sensors_dict if len(target_sensors_dict)>0 else None





