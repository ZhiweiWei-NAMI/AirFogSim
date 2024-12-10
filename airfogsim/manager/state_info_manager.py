import pandas as pd


class StateInfoManager:
    def __init__(self, state_config):
        self._state_config = state_config
        self._time_window = self._state_config.get('time_window', 10) # seconds
        self._current_time = -1
        self._fog_node_state_attributes = self._state_config.get('fog_node_state_attributes', [])
        self._task_node_state_attributes = self._state_config.get('task_node_state_attributes', [])
        self._task_state_attributes = self._state_config.get('task_state_attributes', [])
        self._fog_node_state_df = pd.DataFrame(columns=['node_id', 'time'] + self._fog_node_state_attributes)
        self._task_node_state_df = pd.DataFrame(columns=['node_id', 'time'] + self._task_node_state_attributes)
        self._task_state_df = pd.DataFrame(columns=['task_id', 'time'] + self._task_state_attributes)

    def logNodeState(self, fog_nodes, task_nodes, cur_time):
        self._current_time = cur_time
        # 把cur_time-time_window之前的数据删除
        self._fog_node_state_df = self._fog_node_state_df[self._fog_node_state_df['time'] >= cur_time - self._time_window]
        self._task_node_state_df = self._task_node_state_df[self._task_node_state_df['time'] >= cur_time - self._time_window]
        # 存储fog_node的状态
        for node in fog_nodes:
            node_id = node.getId()
            state = [node_id, cur_time]
            for attr in self._fog_node_state_attributes:
                state.append(getattr(node, '_'+attr))
            self._fog_node_state_df.loc[len(self._fog_node_state_df)] = state
        # 存储task_node的状态
        for node in task_nodes:
            node_id = node.getId()
            state = [node_id, cur_time]
            for attr in self._task_node_state_attributes:
                state.append(getattr(node, '_'+attr))
            self._task_node_state_df.loc[len(self._task_node_state_df)] = state

    def logTaskState(self, tasks, cur_time):
        self._current_time = cur_time
        self._task_state_df = self._task_state_df[self._task_state_df['time'] >= cur_time - self._time_window]
        for task in tasks:
            task_id = task.getTaskId()
            state = [task_id, cur_time]
            for attr in self._task_state_attributes:
                state.append(getattr(task, '_'+attr))
            self._task_state_df.loc[len(self._task_state_df)] = state
