import numpy as np

from .base_sched import BaseScheduler


class SensorScheduler(BaseScheduler):
    @staticmethod
    def getAccuracyById(env, sensor_id):
        env.sensor_manager.getAccuracyById(sensor_id)

    @staticmethod
    def getNodeIdById(env, sensor_id):
        env.sensor_manager.getNodeIdById(sensor_id)

    @staticmethod
    def getHighestAccurateIdleSensorOnUAV(env, type, lowest_accuracy, excluded_sensor_ids):
        candidate_sensors = env.sensor_manager.getSensorsByStateAndType('idle', type)
        appointed_node_id = None
        appointed_sensor_id = None
        appointed_sensor_accuracy = 0

        # Choose the sensor with the highest accuracy among the idle sensors
        for node_id, sensors in candidate_sensors.items():
            for sensor in sensors:
                if sensor.getSensorAccuracy() > max(lowest_accuracy,
                                                    appointed_sensor_accuracy) and sensor.getSensorId() not in excluded_sensor_ids and env._getNodeTypeById(
                        node_id) == 'U':
                    # nonlocal appointed_node_id, appointed_sensor_id, appointed_sensor_accuracy
                    appointed_sensor_accuracy = sensor.getSensorAccuracy()
                    appointed_node_id = node_id
                    appointed_sensor_id = sensor.getSensorId()
        return appointed_node_id, appointed_sensor_id, appointed_sensor_accuracy

    @staticmethod
    def getLowestAccurateIdleSensorOnUAV(env, type, lowest_accuracy, excluded_sensor_ids):
        candidate_sensors = env.sensor_manager.getSensorsByStateAndType('idle', type)
        appointed_node_id = None
        appointed_sensor_id = None
        appointed_sensor_accuracy = 0

        # Choose the sensor with the highest accuracy among the idle sensors
        for node_id, sensors in candidate_sensors.items():
            for sensor in sensors:
                if sensor.getSensorAccuracy() > lowest_accuracy and sensor.getSensorId() not in excluded_sensor_ids and env._getNodeTypeById(
                        node_id) == 'U':
                    if appointed_sensor_accuracy == 0 or sensor.getSensorAccuracy() < appointed_sensor_accuracy:
                        # nonlocal appointed_node_id, appointed_sensor_id, appointed_sensor_accuracy
                        appointed_sensor_accuracy = sensor.getSensorAccuracy()
                        appointed_node_id = node_id
                        appointed_sensor_id = sensor.getSensorId()
        return appointed_node_id, appointed_sensor_id, appointed_sensor_accuracy

    @staticmethod
    def getNearestIdleSensorInNodes(env, sensor_type, lowest_accuracy, target_position, node_infos,
                                    excluded_sensor_ids):
        candidate_sensors = env.sensor_manager.getSensorsByStateAndType('idle', sensor_type)
        appointed_node_id = None
        appointed_sensor_id = None
        appointed_sensor_accuracy = 0
        min_distance = None

        # Choose the sensor with the highest accuracy among the idle sensors
        for node_id, node_info in node_infos.items():
            sensors = candidate_sensors.get(node_id, [])
            node_position = node_info['position']
            distance = np.linalg.norm(np.array(target_position) - np.array(node_position))
            if min_distance is None or distance < min_distance:
                for sensor in sensors:
                    if sensor.getSensorAccuracy() > lowest_accuracy and sensor.getSensorId() not in excluded_sensor_ids:
                        # nonlocal appointed_node_id, appointed_sensor_id, appointed_sensor_accuracy
                        appointed_sensor_accuracy = sensor.getSensorAccuracy()
                        appointed_node_id = node_id
                        appointed_sensor_id = sensor.getSensorId()
        return appointed_node_id, appointed_sensor_id, appointed_sensor_accuracy
