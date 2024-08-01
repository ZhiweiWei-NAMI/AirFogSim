class SimpleNode:
    def __init__(self, id, position_x, position_y, position_z, speed=0, acceleration=0):
        self._id = id
        self._position_x = position_x
        self._position_y = position_y
        self._position_z = position_z
        self._speed = speed
        self._acceleration = acceleration

    def getId(self):
        return self._id
    
    def getPosition(self):
        return (self._position_x, self._position_y, self._position_z)
    
    def getSpeed(self):
        return self._speed
    
    def getAcceleration(self):
        return self._acceleration
    
    def setPosition(self, position_x, position_y, position_z):
        self._position_x = position_x
        self._position_y = position_y
        self._position_z = position_z

    def setSpeed(self, speed):
        self._speed = speed

    def setAcceleration(self, acceleration):
        self._acceleration = acceleration

    def __str__(self):
        return "SimpleNode: id={}, position=({}, {}, {}), speed={}, acceleration={}".format(self._id, self._position_x, self._position_y, self._position_z, self._speed, self._acceleration)