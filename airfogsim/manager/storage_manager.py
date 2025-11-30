"""存储/缓存管理器 - 各节点的缓存管理"""
from collections import OrderedDict


class StorageManager:
    def __init__(self, config_storage):
        self._config = config_storage
        # 每节点缓存: {node_id: OrderedDict{content_id: size_bytes}}  (OrderedDict实现LRU)
        self._caches = {}
        # 每节点容量: {node_id: capacity_bytes}
        self._capacities = {}
        # 统计: {node_id: {'hits': int, 'misses': int}}
        self._stats = {}
        self._init_capacities()

    def _init_capacities(self):
        cap_config = self._config.get('capacity_bytes', {})
        for node_id, cap in cap_config.items():
            self._capacities[node_id] = cap
            self._caches[node_id] = OrderedDict()
            self._stats[node_id] = {'hits': 0, 'misses': 0}

    def reset(self):
        for node_id in self._caches:
            self._caches[node_id] = OrderedDict()
            self._stats[node_id] = {'hits': 0, 'misses': 0}

    def getCapacity(self, node_id):
        return self._capacities.get(node_id, 0)

    def getUsedSize(self, node_id):
        if node_id not in self._caches:
            return 0
        return sum(self._caches[node_id].values())

    def hasContent(self, node_id, content_id):
        if node_id not in self._caches:
            return False
        return content_id in self._caches[node_id]

    def get(self, node_id, content_id):
        """获取内容，命中则返回size并更新LRU顺序，未命中返回None"""
        if node_id not in self._caches:
            return None
        cache = self._caches[node_id]
        if content_id in cache:
            self._stats[node_id]['hits'] += 1
            cache.move_to_end(content_id)
            return cache[content_id]
        self._stats[node_id]['misses'] += 1
        return None

    def put(self, node_id, content_id, size_bytes):
        """放入内容，必要时LRU驱逐"""
        if node_id not in self._caches:
            # 若该节点未配置缓存，默认不存储
            return False
        capacity = self._capacities.get(node_id, 0)
        if size_bytes > capacity:
            return False
        cache = self._caches[node_id]
        # 如果已存在，更新
        if content_id in cache:
            del cache[content_id]
        # LRU驱逐
        while self.getUsedSize(node_id) + size_bytes > capacity and cache:
            cache.popitem(last=False)
        cache[content_id] = size_bytes
        return True

    def evict(self, node_id, content_id):
        """手动驱逐"""
        if node_id in self._caches and content_id in self._caches[node_id]:
            del self._caches[node_id][content_id]
            return True
        return False

    def getHitRatio(self, node_id):
        if node_id not in self._stats:
            return 0.0
        s = self._stats[node_id]
        total = s['hits'] + s['misses']
        if total == 0:
            return 0.0
        return s['hits'] / total

    def getCacheState(self, node_id):
        """返回节点缓存状态"""
        if node_id not in self._caches:
            return {'capacity': 0, 'used': 0, 'items': 0, 'hit_ratio': 0.0}
        return {
            'capacity': self._capacities.get(node_id, 0),
            'used': self.getUsedSize(node_id),
            'items': len(self._caches[node_id]),
            'hit_ratio': self.getHitRatio(node_id)
        }
