from base_sched import BaseScheduler

class SensorScheduler(BaseScheduler):
    @staticmethod
    def getAppointedSensor(env,type,accuracy):
        candidate_sensors= env.sensor_manager.getSensorsByType('idle',type)
        appointed_sensor_accuracy = accuracy
        appointed_node_id=None
        appointed_sensor_id=None
        appointed_sensor_accuracy=0

        # Choose the sensor with the highest accuracy among the idle sensors
        for node_id, sensors in candidate_sensors:
            for sensor in sensors:
                if sensor.getSensorAccuracy() > appointed_sensor_accuracy:
                    nonlocal appointed_node_id, appointed_sensor_id, appointed_sensor_accuracy
                    appointed_sensor_accuracy = sensor.getSensorAccuracy()
                    appointed_node_id = node_id
                    appointed_sensor_id = sensor.getSensorId()
        return appointed_node_id,appointed_sensor_id,appointed_sensor_accuracy