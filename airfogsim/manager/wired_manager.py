"""有线网络管理器 - RSU/Cloud之间的回传链路"""

class WiredNetworkManager:
    def __init__(self, config_wired):
        self._config = config_wired
        # 链路拓扑: {(src, dst): {'capacity_mbps': float, 'prop_ms': float, 'queue_bytes': int}}
        self._links = {}
        # 当前队列状态: {(src, dst): current_queue_bytes}
        self._queues = {}
        # 待传流: {task_id: {'src': str, 'dst': str, 'path': list, 'remaining_bytes': float}}
        self._flows = {}
        self._init_topology()

    def _init_topology(self):
        edges = self._config.get('edges', [])
        for edge in edges:
            u, v = edge['u'], edge['v']
            self._links[(u, v)] = {
                'capacity_mbps': edge.get('capacity_mbps', 100),
                'prop_ms': edge.get('prop_ms', 1.0),
            }
            self._queues[(u, v)] = 0
            # 双向链路
            if edge.get('bidirectional', True):
                self._links[(v, u)] = self._links[(u, v)].copy()
                self._queues[(v, u)] = 0

    def reset(self):
        self._flows = {}
        for key in self._queues:
            self._queues[key] = 0

    def hasLink(self, src, dst):
        """检查两节点间是否有直连有线链路"""
        return (src, dst) in self._links

    def enqueue(self, task_id, src, dst, size_bytes):
        """将一个流加入传输队列"""
        if not self.hasLink(src, dst):
            return False
        self._flows[task_id] = {
            'src': src,
            'dst': dst,
            'remaining_bytes': size_bytes
        }
        self._queues[(src, dst)] += size_bytes
        return True

    def step(self, simulation_interval_s):
        """执行一步传输，返回 {task_id: transmitted_bytes}"""
        results = {}
        # 按链路分组
        link_flows = {}
        for task_id, flow in self._flows.items():
            link = (flow['src'], flow['dst'])
            if link not in link_flows:
                link_flows[link] = []
            link_flows[link].append(task_id)

        # 每条链路公平分配带宽
        for link, task_ids in link_flows.items():
            if link not in self._links:
                continue
            capacity_bytes_per_s = self._links[link]['capacity_mbps'] * 1e6 / 8
            total_capacity = capacity_bytes_per_s * simulation_interval_s
            per_flow_capacity = total_capacity / len(task_ids)

            for task_id in task_ids:
                flow = self._flows[task_id]
                transmitted = min(per_flow_capacity, flow['remaining_bytes'])
                flow['remaining_bytes'] -= transmitted
                self._queues[link] = max(0, self._queues[link] - transmitted)
                results[task_id] = transmitted

        # 移除已完成的流
        done_tasks = [tid for tid, flow in self._flows.items() if flow['remaining_bytes'] <= 0]
        for tid in done_tasks:
            del self._flows[tid]

        return results

    def getFlowRemaining(self, task_id):
        """获取某流的剩余字节数"""
        if task_id in self._flows:
            return self._flows[task_id]['remaining_bytes']
        return 0

    def getLinkUtilization(self):
        """获取各链路利用率"""
        return {link: q for link, q in self._queues.items()}
