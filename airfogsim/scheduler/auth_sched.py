"""
Authentication Scheduler for AirFogSim
完全按照TaskScheduler的设计模式实现，对应SimpleAuthManager
"""

from .base_sched import BaseScheduler
from typing import List, Dict, Optional


class AuthScheduler(BaseScheduler):
    """
    认证调度器 - 完全按照TaskScheduler的设计模式
    
    对应SimpleAuthManager，提供静态方法接口
    """
    
    @staticmethod
    def getAuthenticatedNodes(env, node_ids: List[str]) -> List[str]:
        """
        获取认证通过的节点列表 - 类似TaskScheduler.getWaitingToOffloadTasks
        
        Args:
            env: AirFogSim环境实例
            node_ids: 候选节点ID列表
            
        Returns:
            认证通过的节点ID列表
        """
        if not hasattr(env, 'auth_manager'):
            return node_ids
        
        return env.auth_manager.get_authenticated_nodes(node_ids, env.simulation_time)
    
    @staticmethod
    def isNodeAuthenticated(env, node_id: str) -> bool:
        """
        检查节点认证状态 - 类似TaskScheduler.getTaskById
        
        Args:
            env: AirFogSim环境实例
            node_id: 节点ID
            
        Returns:
            True if节点已认证且有效
        """
        if not hasattr(env, 'auth_manager'):
            return True
        
        return env.auth_manager.is_node_authenticated(node_id, env.simulation_time)
    
    @staticmethod
    def setNodeAuthentication(env, node_id: str, node_info: Dict) -> bool:
        """
        设置节点认证 - 类似TaskScheduler.setTaskOffloading
        
        Args:
            env: AirFogSim环境实例
            node_id: 节点ID
            node_info: 节点信息字典
            
        Returns:
            True if设置成功
        """
        if not hasattr(env, 'auth_manager'):
            return True
        
        return env.auth_manager.register_node(node_id, node_info)
    
    @staticmethod
    def getTrustScore(env, node_id: str) -> float:
        """
        获取节点信任分数 - 类似TaskScheduler.getTaskDelay
        
        Args:
            env: AirFogSim环境实例
            node_id: 节点ID
            
        Returns:
            节点信任分数 (0.0-1.0)
        """
        if not hasattr(env, 'auth_manager'):
            return 1.0
        
        return env.auth_manager.get_node_trust_score(node_id)
    
    @staticmethod
    def getAuthenticationStatistics(env) -> Dict:
        """
        获取认证统计信息 - 类似TaskScheduler.getDoneTaskNum
        
        Args:
            env: AirFogSim环境实例
            
        Returns:
            认证系统统计信息字典
        """
        if not hasattr(env, 'auth_manager'):
            return {
                'total_nodes': 0,
                'authenticated_nodes': 0,
                'malicious_nodes': 0,
                'authentication_rate': 100.0,
                'malicious_detection_rate': 0.0
            }
        
        return env.auth_manager.get_authentication_statistics()
    
    @staticmethod
    def validateTaskResult(env, task_id: str, fog_node_id: str, 
                          expected_result: Optional[Dict] = None,
                          actual_result: Optional[Dict] = None) -> bool:
        """
        验证任务结果 - 类似TaskScheduler.moveTaskToDone
        
        Args:
            env: AirFogSim环境实例
            task_id: 任务ID
            fog_node_id: 处理任务的雾节点ID
            expected_result: 期望结果
            actual_result: 实际结果
            
        Returns:
            True if结果有效
        """
        if not hasattr(env, 'auth_manager'):
            return True
        
        return env.auth_manager.validate_task_result(task_id, fog_node_id, expected_result, actual_result)
    
    @staticmethod
    def isNodeMalicious(env, node_id: str) -> bool:
        """
        检查节点是否为恶意节点 - 类似TaskScheduler.isTaskFailed
        
        Args:
            env: AirFogSim环境实例
            node_id: 节点ID
            
        Returns:
            True if节点是恶意的
        """
        if not hasattr(env, 'auth_manager'):
            return False
        
        if hasattr(env.auth_manager, 'malicious_nodes'):
            return node_id in env.auth_manager.malicious_nodes
        return False
    
    @staticmethod
    def filterMaliciousNodes(env, node_ids: List[str]) -> List[str]:
        """
        过滤恶意节点 - 类似TaskScheduler.getFailedTasks的逆操作
        
        Args:
            env: AirFogSim环境实例
            node_ids: 候选节点ID列表
            
        Returns:
            过滤后的安全节点ID列表
        """
        if not hasattr(env, 'auth_manager'):
            return node_ids
        
        safe_nodes = []
        for node_id in node_ids:
            if not AuthScheduler.isNodeMalicious(env, node_id):
                safe_nodes.append(node_id)
        
        return safe_nodes
    
    @staticmethod
    def removeNode(env, node_id: str):
        """
        移除节点 - 类似TaskScheduler.removeTask
        
        Args:
            env: AirFogSim环境实例
            node_id: 节点ID
        """
        if hasattr(env, 'auth_manager'):
            env.auth_manager.remove_node(node_id)
    
    @staticmethod
    def resetAuthentication(env):
        """
        重置认证系统 - 类似TaskScheduler.reset
        
        Args:
            env: AirFogSim环境实例
        """
        if hasattr(env, 'auth_manager'):
            env.auth_manager.reset()
    
    @staticmethod
    def updateAuthenticationStatus(env):
        """
        更新认证状态 - 类似TaskScheduler.checkTasks (但通常在env内部调用)
        
        Args:
            env: AirFogSim环境实例
        """
        if hasattr(env, 'auth_manager'):
            env.auth_manager.update_authentication_status(env.simulation_time)


# 为了兼容性，提供简短别名 - 类似TaskSched
class AuthSched(AuthScheduler):
    """AuthScheduler的简短别名 - 类似TaskSched"""
    pass
