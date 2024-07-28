from inspect import isfunction, getfullargspec
from ..airfogsim_env import AirFogSimEnv

class ValidateStaticMethods(type):
    def __new__(cls, name, bases, attrs):
        for key, value in attrs.items():
            if isfunction(value):  # 检查是否为函数
                if not isinstance(value, staticmethod):
                    raise TypeError(f"Method {key} must be a static method")
                
                # 检查第一个参数是否为 AirFogSimEnv 类的实例
                args = getfullargspec(value.__func__).args
                if len(args) == 0 or args[0] != 'env' or not hasattr(value.__func__, '__annotations__') or value.__func__.__annotations__.get('env') != AirFogSimEnv:
                    raise TypeError(f"The first argument of method {key} must be 'env: AirFogSimEnv'")
                
                # 检查是否以{set, get, add, is, delete}开头，代表增删改查和判断
                if not key.startswith(('set', 'get', 'add', 'delete', 'is')):
                    raise ValueError(f"Method {key} must start with 'is', 'set', 'get', 'add', or 'delete'")
                
        
        return super().__new__(cls, name, bases, attrs)
    

class BaseScheduler(metaclass=ValidateStaticMethods):
    pass
