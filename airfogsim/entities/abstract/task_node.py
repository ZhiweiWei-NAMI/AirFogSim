from .simple_node import SimpleNode
class TaskNode(SimpleNode):
    def __init__(self, id, position_x=0, position_y=0, position_z=0, speed=0, acceleration=0, task_profile=None):
        super(TaskNode, self).__init__(id, position_x, position_y, position_z, speed, acceleration)
        self._task_profile = task_profile

    def getTaskProfile(self):
        return self._task_profile
    
    def setTaskProfile(self, task_profile):
        self._task_profile = task_profile
