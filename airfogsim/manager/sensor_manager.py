import random
from ..entities.sensor import Sensor


class SensorManager:
    """ Sensor Manager is responsible for generating and deploying sensors and managing the sensor status.
    """
    NODE_TYPE = ['vehicle', 'UAV']
    STATE = ['idle', 'busy', 'unavailable']
    ACCURACY_RANGE = [0.5,0.6,0.7,0.8,0.9, 1.00]

    def __init__(self, config_sensing, traffic_manager):
        assert set(config_sensing['node_type']).issubset(
            SensorManager.NODE_TYPE), 'The node type is not supported. Only support {}'.format(SensorManager.NODE_TYPE)
        self._config_sensing = config_sensing
        self._busy_sensors = {}  # key: node_id, value: list of busy sensors
        self._idle_sensors = {}  # key: node_id, value: list of idle sensors
        self._unavailable_sensors = {}  # key: node_id, value: list of unavailable sensors
        self._sensor_id_counter = 0
        self._sensors_per_node = config_sensing['sensors_per_node']
        self._node_type = config_sensing['node_type']
        self._sensor_type_num = config_sensing['sensor_type_num']
        self._initializeSensors(traffic_manager)

    def reset(self):
        self._busy_sensors = {}  # key: node_id, value: list of busy sensors
        self._idle_sensors = {}  # key: node_id, value: list of idle sensors
        self._unavailable_sensors = {}  # key: node_id, value: list of unavailable sensors
        self._sensor_id_counter = 0


    def __getNewSensorId(self):
        new_id=self._sensor_id_counter
        self._sensor_id_counter += 1
        return new_id

    def _initializeSensors(self, traffic_manager):
        UAV_numbers = traffic_manager.getNumberOfUAVs()
        UAV_infos = traffic_manager.getUAVTrafficInfos()
        vehicle_numbers = traffic_manager.getNumberOfVehicles()
        vehicle_infos = traffic_manager.getVehicleTrafficInfos()

        for UAV_id in UAV_infos.keys():
            self._initializeSensorsForNode(UAV_id)
        for vehicle_id in vehicle_infos.keys():
            self._initializeSensorsForNode(vehicle_id)

    def _initializeSensorsForNode(self, node_id):
        self._idle_sensors[node_id] = self._idle_sensors.get(node_id,[])
        assert len(self._idle_sensors[node_id])==0
        for idx in range(self._sensors_per_node):
            new_sensor_type = 'Sensor_type_' + str(random.randint(1, self._sensor_type_num))
            new_sensor_accuracy = random.choice(SensorManager.ACCURACY_RANGE)  # 随机生成0-1之间的离散精度
            new_sensor_id = 'Sensor_' + str(self.__getNewSensorId())
            new_sensor = Sensor(new_sensor_id, new_sensor_type, new_sensor_accuracy, node_id)
            self._idle_sensors[node_id].append(new_sensor)

    def _getIdleSensorById(self, sensor_id):
        target_sensor = None
        target_node_id = None
        for node_id, sensors in self._idle_sensors.items():
            for sensor in sensors:
                if sensor.getSensorId() == sensor_id:
                    target_sensor = sensor
                    target_node_id = node_id
        return target_node_id, target_sensor

    def _getBusySensorById(self, sensor_id):
        target_sensor = None
        target_node_id = None
        for node_id, sensors in self._busy_sensors.items():
            for sensor in sensors:
                if sensor.getSensorId() == sensor_id:
                    target_sensor = sensor
                    target_node_id = node_id
        return target_node_id, target_sensor

    def _getUnavailableSensorById(self, sensor_id):
        target_sensor = None
        target_node_id = None
        for node_id, sensors in self._unavailable_sensors.items():
            for sensor in sensors:
                if sensor.getSensorId() == sensor_id:
                    target_sensor = sensor
                    target_node_id = node_id
        return target_node_id, target_sensor

    def _addSensor(self, sensors_dict, node_id, sensor):
        sensors_dict[node_id] = sensors_dict.get(node_id, [])
        sensors_dict[node_id].append(sensor)

    def _removeSensor(self, sensors_dict, node_id, sensor_id):
        assert node_id in sensors_dict
        sensors_dict[node_id] = [sensor for sensor in sensors_dict[node_id] if sensor.getSensorId() != sensor_id]

    def initializeSensorsByNodeId(self, node_id):
        self._initializeSensorsForNode(node_id)

    def startUseById(self, sensor_id):
        """start use the sensor.

        Args:
            sensor_id (str): The sensor id.

        Returns:

        Examples:
            sensor_manager.startUseById('Sensor_1')
        """

        node_id, sensor = self._getIdleSensorById(sensor_id)
        assert sensor is not None
        sensor.startUse()
        self._removeSensor(self._idle_sensors, node_id, sensor_id)
        self._addSensor(self._busy_sensors, node_id, sensor)

    def endUseById(self, sensor_id):
        """stop use the sensor.

        Args:
            sensor_id (str): The sensor id.

        Returns:

        Examples:
            sensor_manager.stopUseById('Sensor_1')
        """
        node_id, sensor = self._getBusySensorById(sensor_id)
        assert sensor is not None
        sensor.endUse()
        self._removeSensor(self._busy_sensors, node_id, sensor_id)
        self._addSensor(self._idle_sensors, node_id, sensor)

    def _setUnavailableById(self, sensor_id):
        """make the sensor unavailable.

        Args:
            sensor_id (str): The sensor id.

        Returns:

        """

        node_id, sensor = self._getIdleSensorById(sensor_id)
        if sensor is not None:
            sensor.disable()
            self._removeSensor(self._idle_sensors, node_id, sensor_id)
            self._addSensor(self._unavailable_sensors, node_id, sensor)
            return

        node_id, sensor = self._getBusySensorById(sensor_id)
        if sensor is not None:
            sensor.disable()
            self._removeSensor(self._busy_sensors, node_id, sensor_id)
            self._addSensor(self._unavailable_sensors, node_id, sensor)
            return

        assert sensor is not None, f'{sensor_id} is not exist'

    def getUsableById(self, sensor_id):
        """check if the sensor is usable.

        Args:
            sensor_id (str): The sensor id.

        Returns:

        Examples:
            sensor_manager.getUsableById('Sensor_1')
        """
        _, sensor = self._getIdleSensorById(sensor_id)
        if sensor is not None:
            return sensor.isUsable()

        _, sensor = self._getBusySensorById(sensor_id)
        if sensor is not None:
            return sensor.isUsable()

        assert sensor is not None, f'{sensor_id} is not exist'

    def getSensorsByStateAndType(self, state, type):
        assert state in SensorManager.STATE, 'The state is not supported. Only support {}'.format(SensorManager.STATE)
        sensors_dict = {}
        target_sensors_dict = {}

        if state == 'idle':
            sensors_dict = self._idle_sensors
        elif state == 'busy':
            sensors_dict = self._busy_sensors
        elif state == 'unavailable':
            sensors_dict = self._unavailable_sensors

        for node_id, sensors in sensors_dict.items():
            target_sensors_dict[node_id] = []
            for sensor in sensors:
                if sensor.getSensorType() == type:
                    target_sensors_dict[node_id].append(sensor)
            if len(target_sensors_dict[node_id]) == 0:
                target_sensors_dict.pop(node_id)
        return target_sensors_dict

    def getUsingSensorsNumByNodeId(self, node_id):
        sensor_list = self._busy_sensors.get(node_id, [])
        for sensor in sensor_list:
            assert sensor.isUsing()
        return len(sensor_list)

    def disableByNodeId(self, node_id):
        sensor_list = self._idle_sensors.get(node_id, [])
        for sensor in sensor_list:
            self._setUnavailableById(sensor.getSensorId())

        sensor_list = self._busy_sensors.get(node_id, [])
        for sensor in sensor_list:
            self._setUnavailableById(sensor.getSensorId())

    def getBusySensorsNum(self):
        num = 0
        for node_id, sensors in self._busy_sensors.items():
            num += len(sensors)
        return num

    def getIdleSensorsNum(self):
        num = 0
        for node_id, sensors in self._idle_sensors.items():
            num += len(sensors)
        return num

    def getUnavailableSensorsNum(self):
        num = 0
        for node_id, sensors in self._unavailable_sensors.items():
            num += len(sensors)
        return num

    def getAccuracyById(self, sensor_id):
        for node_id, sensors in self._idle_sensors.items():
            for sensor in sensors:
                if sensor.getSensorId() == sensor_id:
                    return sensor.getSensorAccuracy()
        for node_id, sensors in self._busy_sensors.items():
            for sensor in sensors:
                if sensor.getSensorId() == sensor_id:
                    return sensor.getSensorAccuracy()
        for node_id, sensors in self._unavailable_sensors.items():
            for sensor in sensors:
                if sensor.getSensorId() == sensor_id:
                    return sensor.getSensorAccuracy()

    def getNodeIdById(self, sensor_id):
        for node_id, sensors in self._idle_sensors.items():
            for sensor in sensors:
                if sensor.getSensorId() == sensor_id:
                    return sensor.getDeployedNodeId()
        for node_id, sensors in self._busy_sensors.items():
            for sensor in sensors:
                if sensor.getSensorId() == sensor_id:
                    return sensor.getDeployedNodeId()
        for node_id, sensors in self._unavailable_sensors.items():
            for sensor in sensors:
                if sensor.getSensorId() == sensor_id:
                    return sensor.getDeployedNodeId()

    def getConfig(self, name):
        return self._config_sensing.get(name, None)
