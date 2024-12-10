import pandas as pd


class StateInfoManager:
    def __init__(self, state_config):
        self._state_config = state_config
        self._fog_node_state_attributes = self._state_config.get('fog_node_state_attributes', [])
        self._task_node_state_attributes = self._state_config.get('task_node_state_attributes', [])
        self._task_state_attributes = self._state_config.get('task_state_attributes', [])
        self._fog_node_state_df = pd.DataFrame(columns=['node_id', 'time'] + self._fog_node_state_attributes)
        self._task_node_state_df = pd.DataFrame(columns=['node_id', 'time'] + self._task_node_state_attributes)
        self._task_state_df = pd.DataFrame(columns=['task_id', 'time'] + self._task_state_attributes)