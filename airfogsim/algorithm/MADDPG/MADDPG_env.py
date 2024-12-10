from madrl_environments.pursuit import MAWaterWorld_mod
from .MADDPG_model import MADDPG
import numpy as np
import torch
import visdom
from params import scale_reward

class DDQN_Env:

    def __init__(self,n_agents, dim_obs, dim_act):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # 智能体数量
        self.n_agents=n_agents
        # 观测状态维度
        self.dim_obs = dim_obs
        # 动作空间维度（连续动作，因此每个维度表示一个动作特征）
        self.dim_act = dim_act

        # 超参数
        self.lr = 2e-3  # 学习率
        self.gamma = 0.9  # 折扣因子
        self.buffer_size = 500  # 经验池容量
        self.batch_size = 32  # 每次训练选取的经验数量
        self.episodes_before_train = 10
        self.epsilon = 0.9  # 探索系数
        self.eps_end = 0.01  # 最低探索系数
        self.eps_dec = 5e-7  # 探索系数衰减率
        self.target_update = 200  # 目标网络的参数的更新频率

        self.dim_hidden = 128  # 隐含层神经元个数
        self.train_min_size = 200  # 经验池超过200后再训练(train_min_size>batch_size)
        self.tau = 0.995  # 目标网络软更新平滑因子（策略网络权重）
        self.smooth_factor = 0.995  # 最大q值平滑因子（旧值权重）

        # self.return_list = []  # 记录每次迭代的return，即链上的reward之和
        self.max_q_value = 0  # 最大state_value
        # self.max_q_value_list = []  # 保存所有最大的state_value

        # 模型文件路径
        self.model_base_dir = "./airfogsim/algorithm/DDQN/model/"

        # 实例化 Double-DQN
        self.agent = MADDPG(self.n_agents, self.dim_obs, self.dim_act,self.lr, self.gamma, self.buffer_size, self.batch_size,
                 self.episodes_before_train, self.train_min_size, self.tau, self.device)

    def takeAction(self, state, mask):
        # 状态state时做动作选择，action为动作索引
        is_random, max_q_value, action = self.agent.take_action(state,mask)
        # 平滑处理最大state_value
        self.max_q_value = max_q_value * (1 - self.smooth_factor) + self.max_q_value * self.smooth_factor
        # 保存每次迭代的最大state_value
        # self.max_q_value_list.append(self.max_q_value)
        return is_random,self.max_q_value,action

    def addExperience(self, state, action,mask, reward, next_state,next_mask, done):
        # 添加经验池
        self.agent.remember(state, action,mask, reward, next_state,next_mask, done)

    def train(self):
        self.agent.update()

    def saveModel(self,episode):
        self.agent.save_models(episode,self.model_base_dir)

    def loadModel(self,episode):
        self.agent.load_models(episode,self.model_base_dir)

# do not render the scene
e_render = False

food_reward = 10.
poison_reward = -1.
encounter_reward = 0.01
n_coop = 2
world = MAWaterWorld_mod(n_pursuers=2, n_evaders=50,
                         n_poison=50, obstacle_radius=0.04,
                         food_reward=food_reward,
                         poison_reward=poison_reward,
                         encounter_reward=encounter_reward,
                         n_coop=n_coop,
                         sensor_range=0.2, obstacle_loc=None, )

vis = visdom.Visdom(port=5274)
reward_record = []

np.random.seed(1234)
th.manual_seed(1234)
world.seed(1234)
n_agents = world.n_pursuers
n_states = 213
n_actions = 2
capacity = 1000000
batch_size = 1000
gamma = 0.95
tau = 0.01

n_episode = 20000
max_steps = 1000
episodes_before_train = 100

win = None
param = None

maddpg = MADDPG(n_agents, n_states, n_actions, batch_size, capacity,
                episodes_before_train)

FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor
for i_episode in range(n_episode):
    obs = world.reset()
    obs = np.stack(obs)
    if isinstance(obs, np.ndarray):
        obs = th.from_numpy(obs).float()
    total_reward = 0.0
    rr = np.zeros((n_agents,))
    for t in range(max_steps):
        # render every 100 episodes to speed up training
        if i_episode % 100 == 0 and e_render:
            world.render()
        obs = obs.type(FloatTensor)
        action = maddpg.select_action(obs).data.cpu()
        obs_, reward, done, _ = world.step(action.numpy())

        reward = th.FloatTensor(reward).type(FloatTensor)
        obs_ = np.stack(obs_)
        obs_ = th.from_numpy(obs_).float()
        if t != max_steps - 1:
            next_obs = obs_
        else:
            next_obs = None

        total_reward += reward.sum()
        rr += reward.cpu().numpy()
        maddpg.memory.push(obs.data, action, next_obs, reward)
        obs = next_obs

        c_loss, a_loss = maddpg.update_policy()
    maddpg.episode_done += 1
    print('Episode: %d, reward = %f' % (i_episode, total_reward))
    reward_record.append(total_reward)

    if maddpg.episode_done == maddpg.episodes_before_train:
        print('training now begins...')
        print('MADDPG on WaterWorld\n' +
              'scale_reward=%f\n' % scale_reward +
              'agent=%d' % n_agents +
              ', coop=%d' % n_coop +
              ' \nlr=0.001, 0.0001, sensor_range=0.3\n' +
              'food=%f, poison=%f, encounter=%f' % (
                  food_reward,
                  poison_reward,
                  encounter_reward))

    if win is None:
        win = vis.line(X=np.arange(i_episode, i_episode+1),
                       Y=np.array([
                           np.append(total_reward, rr)]),
                       opts=dict(
                           ylabel='Reward',
                           xlabel='Episode',
                           title='MADDPG on WaterWorld_mod\n' +
                           'agent=%d' % n_agents +
                           ', coop=%d' % n_coop +
                           ', sensor_range=0.2\n' +
                           'food=%f, poison=%f, encounter=%f' % (
                               food_reward,
                               poison_reward,
                               encounter_reward),
                           legend=['Total'] +
                           ['Agent-%d' % i for i in range(n_agents)]))
    else:
        vis.line(X=np.array(
            [np.array(i_episode).repeat(n_agents+1)]),
                 Y=np.array([np.append(total_reward,
                                       rr)]),
                 win=win,
                 update='append')
    if param is None:
        param = vis.line(X=np.arange(i_episode, i_episode+1),
                         Y=np.array([maddpg.var[0]]),
                         opts=dict(
                             ylabel='Var',
                             xlabel='Episode',
                             title='MADDPG on WaterWorld: Exploration',
                             legend=['Variance']))
    else:
        vis.line(X=np.array([i_episode]),
                 Y=np.array([maddpg.var[0]]),
                 win=param,
                 update='append')

world.close()