# AirFogSim AI 开发提示词（最小指引）

目的：在不侵入现有工作流的前提下，快速实现和评测新算法（启发式/学习算法）。

## 基本准则
- 只修改算法相关代码，不改动环境核心逻辑。
- 不使用过度的 try/except；遇到报错直接修复根因。
- 遵循模块边界：通过 Scheduler/Manager API 与环境交互。
- 能复现实验：固定随机种子，记录关键指标。

## 代码位置与入口
- 算法基类：`airfogsim/airfogsim_algorithm.py`
  - 继承 `BaseAlgorithmModule`，可重写：
    - `initialize(env)`：设置任务生成、奖励等。
    - `scheduleStep(env)`：每步决策（返回/卸载/通信/计算/任务/交通）。
- 示例脚本：`examples/example01_offloading_example.py`
- 可选自定义算法目录：`airfogsim/algorithm/<YourAlgo>/`（若需要专用网络/模型）

## 推荐使用的接口（只列最小必要）
- 调度入口：`from airfogsim import AirFogSimScheduler as Sched`
  - 任务：`Sched.getTaskScheduler()`（获取待卸载任务、生成/设置返回路径等）
  - 通信：`Sched.getCommunicationScheduler()`（`getNumberOfRB()`、`setCommunicationWithRB(...)`）
  - 计算：`Sched.getComputationScheduler()`（按需获取计算相关接口）
  - 任务群/传感器/交通：`getMissionScheduler()`、`getSensorScheduler()`、`getTrafficScheduler()`
  - 奖励：`Sched.getRewardScheduler()`（`setModel(env, 'REWARD', expr)`）
- 计算资源分配回调（可选）：在 `env.alloc_cpu_callback` 注册函数，签名与现有用法保持一致。
- 有线与存储（可选，已最小实现）：
  - 配置见 `examples/config.yaml` 的 `wired` 与 `storage` 段；未配置则不生效。

## 最小算法骨架（示意）
```python
from airfogsim import AirFogSimScheduler as Sched
from airfogsim.airfogsim_algorithm import BaseAlgorithmModule

class MyAlgo(BaseAlgorithmModule):
    def initialize(self, env):
        super().initialize(env)
        Sched.getRewardScheduler().setModel(env, 'REWARD', '1/task_delay')

    def scheduleStep(self, env):
        # 1) 为待卸载任务分配RB（示例：每任务1个RB）
        task_infos = Sched.getTaskScheduler().getAllToOffloadTaskInfos(env)
        for t in task_infos:
            Sched.getCommunicationScheduler().setCommunicationWithRB(env, t['task_id'], [0])
        # 2) 其余调度按基类流程
        super().scheduleStep(env)
```

## 开发建议
- 指标：使用 `RewardScheduler` 可按表达式快速定义奖励；`StateInfoManager` 可开启 `log_state` 记录。
- 调参：在 `examples/config.yaml` 中修改任务、通信、能量、wired/storage 配置。
- 调试：`env.render()` 可视化；打印 `TaskScheduler.getDoneTaskNum(env)` 等快速 sanity-check。

## 你可以做什么
- 启发式：最近邻/最短路/简单打分（任务价值、时延、能耗）
- 学习算法：策略网络仅读状态、输出 RB/路由/分配；通过 Scheduler/回调落地决策。

（结束）
