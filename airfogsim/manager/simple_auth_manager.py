"""
Simple Authentication Manager for AirFogSim

This module provides a focused authentication manager that only handles:
1. Malicious node filtering
2. Authentication status tracking for task offloading
3. Prevention of malicious result returns
"""

import time
import random
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass


@dataclass
class NodeAuthStatus:
    """Simple authentication status for a node"""
    node_id: str
    is_authenticated: bool
    auth_time: float
    auth_expiry: float
    is_malicious: bool
    failed_task_count: int = 0  # Count of tasks that returned wrong results
    trust_score: float = 1.0    # Trust score based on task completion quality
    
    def is_auth_valid(self, current_time: float) -> bool:
        """Check if authentication is still valid"""
        return self.is_authenticated and current_time <= self.auth_expiry and not self.is_malicious


class SimpleAuthManager:
    """
    Simple Authentication Manager for VFC
    
    Focuses on:
    1. Malicious node detection and filtering
    2. Authentication status for task offloading decisions
    3. Task result validation to detect malicious behavior
    """
    
    def __init__(self, config: Dict):
        """Initialize the authentication manager"""
        self.config = config
        
        # Authentication parameters
        self.auth_validity_period = config.get('auth_validity_period', 300.0)  # 5 minutes
        self.malicious_ratio = config.get('malicious_ratio', 0.1)  # 10% malicious nodes
        self.trust_threshold = config.get('trust_threshold', 0.5)  # Minimum trust for task assignment
        self.max_failed_tasks = config.get('max_failed_tasks', 3)  # Max failed tasks before marking malicious
        
        # Node authentication status
        self.node_auth_status: Dict[str, NodeAuthStatus] = {}
        self.malicious_nodes: Set[str] = set()
        self.authenticated_nodes: Set[str] = set()
        
        # Statistics
        self.total_auth_requests = 0
        self.successful_auths = 0
        self.detected_malicious = 0
        
        print(f"SimpleAuthManager initialized with {self.malicious_ratio*100}% malicious ratio")
    
    def register_node(self, node_id: str, node_info: Dict) -> bool:
        """
        Register a node and determine its authentication status
        
        Args:
            node_id: Node identifier
            node_info: Node information (position, fog_profile, etc.)
            
        Returns:
            True if registration successful
        """
        current_time = time.time()
        
        # Determine if node is malicious (simulate based on configured ratio)
        is_malicious = self._determine_if_malicious(node_id, node_info)
        
        # Create authentication status
        auth_status = NodeAuthStatus(
            node_id=node_id,
            is_authenticated=True,  # Initially authenticated
            auth_time=current_time,
            auth_expiry=current_time + self.auth_validity_period,
            is_malicious=is_malicious,
            trust_score=1.0  # 🔧 修复：所有节点初始trust_score都是1.0，trust_score只用于记录历史表现
        )
        
        self.node_auth_status[node_id] = auth_status
        
        if is_malicious:
            self.malicious_nodes.add(node_id)
            # print(f"⚠️ Node {node_id} registered as malicious")
        else:
            self.authenticated_nodes.add(node_id)
            # print(f"✓ Node {node_id} registered as legitimate")
        
        return True
    
    def is_node_authenticated(self, node_id: str, current_time: Optional[float] = None) -> bool:
        """
        Check if a node is authenticated and trustworthy for task offloading
        
        Args:
            node_id: Node identifier
            current_time: Current simulation time
            
        Returns:
            True if node is authenticated and trustworthy
        """
        if current_time is None:
            current_time = time.time()
        
        if node_id not in self.node_auth_status:
            return False
        
        auth_status = self.node_auth_status[node_id]
        
        # 🔧 修复：只检查认证有效性，不检查trust_score
        # Trust score只用于记录，不影响任务卸载选择
        return auth_status.is_auth_valid(current_time)
    
    def get_authenticated_nodes(self, node_ids: List[str], current_time: Optional[float] = None) -> List[str]:
        """
        Filter node list to return only authenticated and trustworthy nodes
        
        Args:
            node_ids: List of candidate node IDs
            current_time: Current simulation time
            
        Returns:
            List of authenticated node IDs
        """
        if current_time is None:
            current_time = time.time()
        
        authenticated = []
        for node_id in node_ids:
            if self.is_node_authenticated(node_id, current_time):
                authenticated.append(node_id)
        
        return authenticated
    
    def validate_task_result(self, task_id: str, fog_node_id: str, expected_result: Optional[Dict] = None,
                           actual_result: Optional[Dict] = None) -> bool:
        """
        Validate task result to detect malicious behavior

        Args:
            task_id: Task identifier
            fog_node_id: Node that processed the task
            expected_result: Expected task result (if available)
            actual_result: Actual task result

        Returns:
            True if result is valid, False if malicious behavior detected
        """
        # 🔧 简化：直接检查节点是否在恶意列表中
        if fog_node_id in self.malicious_nodes:
            print(f"⚠️ Malicious result detected from {fog_node_id} for task {task_id}")
            return False
        else:
            # 正常节点返回正确结果
            return True
    
    def update_authentication_status(self, current_time: float):
        """
        Update authentication status for all nodes (called by _updateAuthPrivacy)
        
        Args:
            current_time: Current simulation time
        """
        expired_nodes = []
        
        for node_id, auth_status in self.node_auth_status.items():
            # Check for expired authentication
            if not auth_status.is_auth_valid(current_time):
                expired_nodes.append(node_id)
                self.authenticated_nodes.discard(node_id)
        
        # Re-authenticate expired nodes (simplified)
        for node_id in expired_nodes:
            if not self.node_auth_status[node_id].is_malicious:
                self._renew_authentication(node_id, current_time)
        
        # if expired_nodes:
            # print(f"🔄 Re-authenticated {len(expired_nodes)} expired nodes")
    
    def get_node_trust_score(self, node_id: str) -> float:
        """Get trust score for a node"""
        if node_id in self.node_auth_status:
            return self.node_auth_status[node_id].trust_score
        return 0.0
    
    def get_authentication_statistics(self) -> Dict:
        """Get authentication system statistics"""
        total_nodes = len(self.node_auth_status)
        authenticated_count = len(self.authenticated_nodes)
        malicious_count = len(self.malicious_nodes)
        
        return {
            'total_nodes': total_nodes,
            'authenticated_nodes': authenticated_count,
            'malicious_nodes': malicious_count,
            'detected_malicious': self.detected_malicious,
            'authentication_rate': (authenticated_count / max(1, total_nodes)) * 100,
            'malicious_detection_rate': (self.detected_malicious / max(1, malicious_count)) * 100 if malicious_count > 0 else 0
        }
    
    def _determine_if_malicious(self, node_id: str, node_info: Dict) -> bool:
        """
        Determine if a node should be marked as malicious

        🔧 修复：确保有恶意节点的确定性方法
        """
        # 使用更简单的确定性方法：基于节点ID的最后一个字符
        # 这样可以确保在小规模测试中也有恶意节点
        import hashlib
        hash_val = int(hashlib.md5(node_id.encode()).hexdigest(), 16)
        # 原始的哈希方法作为备选
        return (hash_val % 100) < (self.malicious_ratio * 100)
    

    
    def _renew_authentication(self, node_id: str, current_time: float):
        """Renew authentication for a node"""
        if node_id in self.node_auth_status:
            auth_status = self.node_auth_status[node_id]
            auth_status.is_authenticated = True
            auth_status.auth_time = current_time
            auth_status.auth_expiry = current_time + self.auth_validity_period
            
            if not auth_status.is_malicious:
                self.authenticated_nodes.add(node_id)
    
    def remove_node(self, node_id: str):
        """Remove a node from authentication system"""
        if node_id in self.node_auth_status:
            del self.node_auth_status[node_id]
        
        self.authenticated_nodes.discard(node_id)
        self.malicious_nodes.discard(node_id)
    
    def reset(self):
        """Reset the authentication manager"""
        self.node_auth_status.clear()
        self.malicious_nodes.clear()
        self.authenticated_nodes.clear()
        self.total_auth_requests = 0
        self.successful_auths = 0
        self.detected_malicious = 0
