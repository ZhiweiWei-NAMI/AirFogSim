from base_sched import BaseScheduler

class TrafficScheduler(BaseScheduler):
    @staticmethod
    def getCurrentTime(env):
        return env.traffic_manager.getCurrentTime()