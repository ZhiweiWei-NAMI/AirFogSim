from inspect import isfunction, getfullargspec

class AirFogSimEnv:
    pass

class ValidateStaticMethods(type):
    def __new__(cls, name, bases, attrs):
        for key, value in attrs.items():
            # 确保方法是静态方法
            if isfunction(value) and isinstance(value, staticmethod):
                # 获取函数参数列表
                args = getfullargspec(value.__func__).args
                annotations = value.__func__.__annotations__
                
                # 检查第一个参数
                if not args or args[0] != 'env' or 'env' not in annotations or annotations['env'] != AirFogSimEnv:
                    raise TypeError(f"The first argument of method {key} must be 'env' of type AirFogSimEnv")

                # 检查方法名称是否以指定动词开头
                if not key.startswith(('set', 'get', 'add', 'delete')):
                    raise ValueError(f"Method {key} must start with 'set', 'get', 'add', or 'delete'")

        return super().__new__(cls, name, bases, attrs)

class BaseScheduler(metaclass=ValidateStaticMethods):
    pass

class ComputationScheduler(BaseScheduler):
    @staticmethod
    def setCPUbyFogNodeName(fog_node_name: str, resource_allocation: list):
        print("This should not print if the first argument is not 'env' of type AirFogSimEnv.")
# Test the code
# This should raise an error
scheduler = ComputationScheduler()