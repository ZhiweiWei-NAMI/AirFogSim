# import torch
# import numpy as np
# import gym
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# from parsers import args
# from DDQN_model import ReplayBuffer, Double_DQN
#
# # GPU运算
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#
# # ------------------------------- #
# # （1）加载环境
# # ------------------------------- #
#
# env = gym.make("Pendulum-v1", render_mode="human")
# n_states = env.observation_space.shape[0]  # 状态数 3
# act_low = env.action_space.low  # 最小动作力矩 -2
# act_high = env.action_space.high  # 最大动作力矩 +2
# n_actions = 11  # 动作是连续的[-2,2]，将其离散成11个动作
#
#
# # 确定离散动作区间后，确定其连续动作
# def dis_to_con(discrete_action, n_actions):
#     # discrete_action代表动作索引
#     return act_low + (act_high - act_low) * (discrete_action / (n_actions - 1))
#
#
# # 实例化经验池
# replay_buffer = ReplayBuffer(args.capacity)
#
# # 实例化 Double-DQN
# agent = Double_DQN(n_states,
#                    args.n_hiddens,
#                    n_actions,
#                    args.lr,
#                    args.gamma,
#                    args.epsilon,
#                    args.target_update,
#                    device
#                    )
#
# # ------------------------------- #
# # （2）模型训练
# # ------------------------------- #
#
# return_list = []  # 记录每次迭代的return，即链上的reward之和
# max_q_value = 0  # 最大state_value
# max_q_value_list = []  # 保存所有最大的state_value
#
# for i in range(10):  # 训练几个回合
#     done = False  # 初始，未到达终点
#     state = env.reset()[0]  # 重置环境
#     episode_return = 0  # 记录每回合的return
#
#     with tqdm(total=10, desc='Iteration %d' % i) as pbar:
#
#         while True:
#             # 状态state时做动作选择，返回索引
#             action = agent.take_action(state)
#             # 平滑处理最大state_value
#             max_q_value = agent.max_q_value(state) * 0.005 + \
#                           max_q_value * 0.995
#             # 保存每次迭代的最大state_value
#             max_q_value_list.append(max_q_value)
#             # 将action的离散索引连续化
#             action_continuous = dis_to_con(action, n_actions)
#             # 环境更新
#             next_state, reward, done, _, _ = env.step(action_continuous)
#             # 添加经验池
#             replay_buffer.add(state, action, reward, next_state, done)
#             # 更新状态
#             state = next_state
#             # 更新每回合的回报
#             episode_return += reward
#
#             # 如果经验池超数量过阈值时开始训练
#             if replay_buffer.size() > args.min_size:
#                 # 在经验池中随机抽样batch组数据
#                 s, a, r, ns, d = replay_buffer.sample(args.batch_size)
#                 # 构造训练集
#                 transitions_dict = {
#                     'states': s,
#                     'actions': a,
#                     'next_states': ns,
#                     'rewards': r,
#                     'dones': d,
#                 }
#                 # 模型训练
#                 agent.update(transitions_dict)
#
#             # 到达终点就停止
#             if done is True: break
#
#         # 保存每回合的return
#         return_list.append(episode_return)
#
#         pbar.set_postfix({
#             'step':
#                 agent.count,
#             'return':
#                 '%.3f' % np.mean(return_list[-10:])
#         })
#         pbar.update(1)
#
# # ------------------------------- #
# # （3）绘图
# # ------------------------------- #
#
# plt.subplot(121)
# plt.plot(return_list)
# plt.title('return')
# plt.subplot(122)
# plt.plot(max_q_value_list)
# plt.title('max_q_value')
# plt.show()