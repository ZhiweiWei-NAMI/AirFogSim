import numpy as np
class V2IChannel: 
    # Simulator of the V2I channels
    def __init__(self, n_Veh, n_BS, n_RB, BS_positions):
        '''V2I只存在于一个BS范围内的V2I channel'''
        self.h_bs = 25 # 基站高度25m
        self.h_ms = 1.5 
        self.Decorrelation_distance = 50        
        self.BS_positions = BS_positions 
        self.shadow_std = 8
        self.n_Veh = n_Veh
        self.n_BS = n_BS
        self.n_RB = n_RB
        self.Shadow = np.random.normal(0, self.shadow_std, size=(self.n_Veh, self.n_BS))
        self.update_shadow([])

    
    def remove_vehicle_shadow(self, vid, vid_index):
        '''删除车辆，删除车辆的阴影'''
        index = vid_index[vid]
        self.Shadow = np.delete(self.Shadow, index, axis=0)
        self.n_Veh -= 1
        

    def add_vehicle_shadow(self):
        '''增加车辆，增加车辆的阴影'''
        new_shadow = np.random.normal(0, self.shadow_std, size=(1, self.n_BS))
        self.Shadow = np.concatenate((self.Shadow, new_shadow), axis=0)
        # 更新n_Veh
        self.n_Veh += 1

    def update_positions(self, veh_positions):
        # 把字典转换成列表，通过vid_index来根据index排序
        self.positions = veh_positions
        
    @staticmethod
    def calculate_path_loss(distance, h_bs = 25, h_ms = 1.5):
        distance_3D = np.sqrt(distance ** 2 + (h_bs - h_ms) ** 2)
        PL = 128.1 + 37.6 * np.log10(distance_3D / 1000)  # 根据3GPP，距离的单位是km
        return PL

    def update_pathloss(self):
        if self.n_Veh == 0:
            return
        positions = np.array(self.positions)
        BS_positions = np.array(self.BS_positions)
        
        d1_matrix = np.abs(np.repeat(positions[:, np.newaxis, 0], self.n_BS, axis = 1) - BS_positions[:, 0])
        d2_matrix = np.abs(np.repeat(positions[:, np.newaxis, 1], self.n_BS, axis = 1) - BS_positions[:, 1])
        distance_matrix = np.hypot(d1_matrix, d2_matrix) + 0.001
        
        distance_3D = np.sqrt(distance_matrix ** 2 + (self.h_bs - self.h_ms) ** 2)
        self.PathLoss = 128.1 + 37.6 * np.log10(distance_3D / 1000)  # 根据3GPP，距离的单位是km

    def update_shadow(self, delta_distance_list):
        if len(self.Shadow) != len(delta_distance_list):
            # 1 如果过去一个时间片的车辆数量发生了变化
            Shadow = np.random.normal(0, self.shadow_std, size=(self.n_Veh, self.n_BS))
            self.Shadow = Shadow
        if len(delta_distance_list) != 0:
            delta_distance = np.repeat(delta_distance_list[:,np.newaxis], self.n_BS, axis=1)
            self.Shadow = 10 * np.log10(np.exp(-1*(delta_distance/self.Decorrelation_distance))* (10 ** (self.Shadow / 10)) + np.sqrt(1-np.exp(-2*(delta_distance/self.Decorrelation_distance)))*(10**(np.random.normal(0,self.shadow_std, size=(self.n_Veh, self.n_BS))/10)))

    def update_fast_fading(self):
        gaussian1 = np.random.normal(size=(self.n_Veh, self.n_BS, self.n_RB))
        gaussian2 = np.random.normal(size=(self.n_Veh, self.n_BS, self.n_RB))
        # 计算瑞利分布的信道增益
        r = np.sqrt(gaussian1 ** 2 + gaussian2 ** 2)
        # 计算信道增益的平均值
        omega = np.mean(r ** 2)
        # 计算瑞利分布的概率密度函数
        p_r = (2 * r / omega) * np.exp(-r ** 2 / omega)
        # 转换为 dB 单位
        self.FastFading = 10 * np.log10(p_r)
import math
class V2VChannel:
    def __init__(self, n_Veh, n_RB):
        '''RB数量不变，车辆数量会变，参数设置来源3GPP TR36.885-A.1.4-1'''
        self.t = 0
        self.h_bs = 1.5 # 车作为BS的高度 
        self.h_ms = 1.5 # 车作为MS的高度
        self.fc = 2 # carrier frequency
        self.decorrelation_distance = 10
        self.shadow_std = 3 # shadow的标准值
        self.n_Veh = n_Veh # 车辆数量
        self.n_RB = n_RB # RB数量
        self.Shadow = np.random.normal(0, self.shadow_std, size=(self.n_Veh, self.n_Veh))
        self.update_shadow([])

    def update_positions(self, positions):
        # 把字典转换成列表，通过vid_index来根据index排序
        self.positions = positions

    def update_pathloss(self):
        self.update_pathloss_matrix()
    

    # 考虑到车辆数量变动，需要更新上一时刻的阴影，删除的车辆阴影删除，新增的车辆阴影增加
    def remove_vehicle_shadow(self, vid, vid_index):
        '''删除车辆，删除车辆的阴影'''
        index = vid_index[vid]
        self.Shadow = np.delete(self.Shadow, index, axis=0)
        self.Shadow = np.delete(self.Shadow, index, axis=1)
        self.n_Veh -= 1

    def add_vehicle_shadow(self):
        '''增加车辆，增加车辆的阴影'''
        new_shadow = np.random.normal(0, self.shadow_std, size=(1, self.n_Veh))
        self.Shadow = np.concatenate((self.Shadow, new_shadow), axis=0)
        new_shadow = np.random.normal(0, self.shadow_std, size=(self.n_Veh + 1, 1))
        self.Shadow = np.concatenate((self.Shadow, new_shadow), axis=1)
        self.n_Veh += 1

    def update_shadow(self, delta_distance_list):
        '''输入距离变化，计算阴影变化，基于3GPP的规范'''
        if len(self.Shadow) == 0:
            self.Shadow = np.random.normal(0, self.shadow_std, size=(self.n_Veh, self.n_Veh))
        if len(self.Shadow) != len(delta_distance_list):
            # 1 如果过去一个时间片的车辆数量发生了变化（只会增加）
            Shadow = np.random.normal(0, self.shadow_std, size=(self.n_Veh, self.n_Veh))
            self.Shadow = Shadow
        delta_distance = np.zeros((len(delta_distance_list), len(delta_distance_list)))
        delta_distance_list = np.array(delta_distance_list)
        delta_distance = np.add.outer(delta_distance_list, delta_distance_list)
        if len(delta_distance_list) != 0: 
            exp_term = np.exp(-delta_distance / self.decorrelation_distance)
            sqrt_term = np.sqrt(1 - exp_term**2)
            random_term = np.random.normal(0, self.shadow_std, size=(self.n_Veh, self.n_Veh))
            linear_value1 = 10 ** (self.Shadow / 10)
            linear_value2 = 10 ** (random_term / 10)
            shadow_linear = exp_term * linear_value1 + sqrt_term * linear_value2
            self.Shadow = 10 * np.log10(shadow_linear)
        np.fill_diagonal(self.Shadow, 0)

    def update_fast_fading(self):
        # 生成两个独立的高斯随机变量矩阵
        gaussian1 = np.random.normal(size=(self.n_Veh, self.n_Veh, self.n_RB))
        gaussian2 = np.random.normal(size=(self.n_Veh, self.n_Veh, self.n_RB))
        # 计算瑞利分布的信道增益
        r = np.sqrt(gaussian1 ** 2 + gaussian2 ** 2)
        # 计算信道增益的平均值
        omega = np.mean(r ** 2)
        # 计算瑞利分布的概率密度函数
        p_r = (2 * r / omega) * np.exp(-r ** 2 / omega)
        # 转换为 dB 单位
        self.FastFading = 10 * np.log10(p_r)
        # 对每一个RB, 将对角线的值设置为0
        for i in range(self.n_RB):
            np.fill_diagonal(self.FastFading[:, :, i], 0)

    def get_path_loss_vectorized(self, d, d1, d2):
        if d.shape[0] == 1:
            PL = np.zeros((1, d.shape[1]))
            return PL

        d_bp = 4 * (self.h_bs - 1) * (self.h_ms - 1) * self.fc * (10**9)/(3*10**8)
        
        PL_Los = np.where(d <= 3, 22.7 * np.log10(3) + 41 + 20*np.log10(self.fc/5),
                        np.where(d < d_bp, 22.7 * np.log10(d) + 41 + 20 * np.log10(self.fc/5),
                                40.0 * np.log10(d) + 9.45 - 17.3 * np.log10(self.h_bs) - 17.3 * np.log10(self.h_ms) + 2.7 * np.log10(self.fc/5)))
        
        n_j = np.maximum(2.8 - 0.0024 * d2, 1.84)
        PL_NLos_d1 = PL_Los + 20 - 12.5 * n_j + 10 * n_j * np.log10(np.maximum(d2, 1e-9)) + 3 * np.log10(self.fc/5)
        PL_NLos_d2 = PL_Los + 20 - 12.5 * n_j + 10 * n_j * np.log10(np.maximum(d2, 1e-9)) + 3 * np.log10(self.fc/5)

        PL = np.where(np.minimum(d1, d2) < 10, PL_Los, np.minimum(PL_NLos_d1, PL_NLos_d2))
        
        return PL

    def update_pathloss_matrix(self):
        if self.n_Veh == 0:
            return
        positions = np.array(self.positions)
        d1_matrix = np.abs(np.repeat(positions[:, np.newaxis, 0], self.n_Veh, axis = 1) - np.repeat(positions[np.newaxis, :, 0], self.n_Veh, axis = 0))
        d2_matrix = np.abs(np.repeat(positions[:, np.newaxis, 1], self.n_Veh, axis = 1) - np.repeat(positions[np.newaxis, :, 1], self.n_Veh, axis = 0))
        d_matrix = np.hypot(d1_matrix, d2_matrix) + 0.001

        self.PathLoss = self.get_path_loss_vectorized(d_matrix, d1_matrix, d2_matrix)
        np.fill_diagonal(self.PathLoss, 0)
    def get_path_loss(self, position_A, position_B):
        '''出自IST-4-027756 WINNER II D1.1.2 V1.2 WINNER II的LoS和NLoS模型'''
        d1 = abs(position_A[0] - position_B[0]) # 单位是km
        d2 = abs(position_A[1] - position_B[1])
        d = math.hypot(d1,d2)+0.001
        d_bp = 4 * (self.h_bs - 1) * (self.h_ms - 1) * self.fc * (10**9)/(3*10**8)     
        def PL_Los(d):
            if d <= 3:
                return 22.7 * np.log10(3) + 41 + 20*np.log10(self.fc/5)
            else:
                if d < d_bp:
                    return 22.7 * np.log10(d) + 41 + 20 * np.log10(self.fc/5)
                else:
                    return 40.0 * np.log10(d) + 9.45 - 17.3 * np.log10(self.h_bs) - 17.3 * np.log10(self.h_ms) + 2.7 * np.log10(self.fc/5)
        def PL_NLos(d_a,d_b):
            n_j = max(2.8 - 0.0024*d_b, 1.84)
            return PL_Los(d_a) + 20 - 12.5*n_j + 10 * n_j * np.log10(d_b) + 3*np.log10(self.fc/5)
        if min(d1,d2) < 10: # 以10m作为LoS存在的标准
            PL = PL_Los(d)
            self.ifLOS = True
            self.shadow_std = 3
        else:
            PL = min(PL_NLos(d1,d2), PL_NLos(d2,d1))
            self.ifLOS = False
            self.shadow_std = 4                      # if Non line of sight, the std is 4
        return PL
class V2UChannel:
    def __init__(self, n_Veh, n_RB, n_UAV, hei_UAV):
        '''多个vehicle和多个UAV之间的通信信道'''
        self.t = 0
        self.h_bs = hei_UAV
        self.h_ms = 1.5 # 车作为MS的高度
        self.fc = 2 # carrier frequency
        self.decorrelation_distance = 50
        self.shadow_std = 8 # shadow的标准值
        self.n_Veh = n_Veh # 车辆数量
        self.n_UAV = n_UAV # 无人机数量
        self.n_RB = n_RB # RB数量
        self.Shadow = np.random.normal(0, self.shadow_std, size=(self.n_Veh, self.n_UAV))
        self.update_shadow([], [])
    
    
    def remove_vehicle_shadow(self, vid, vid_index):
        '''删除车辆，删除车辆的阴影'''
        index = vid_index[vid]
        self.Shadow = np.delete(self.Shadow, index, axis=0)
        self.n_Veh -= 1
        
    def add_vehicle_shadow(self):
        '''增加车辆，增加车辆的阴影'''
        new_shadow = np.random.normal(0, self.shadow_std, size=(1, self.n_UAV))
        self.Shadow = np.concatenate((self.Shadow, new_shadow), axis=0)
        # 更新n_Veh
        self.n_Veh += 1



    def update_positions(self, veh_positions, uav_positions):
        '''更新车辆和无人机的位置'''
        self.veh_positions = veh_positions
        self.uav_positions = uav_positions
        
    def update_pathloss(self):
        if self.n_Veh == 0:
            return
        self.PathLoss = self.get_path_loss_matrix(self.veh_positions, self.uav_positions)
        # self.PathLoss = np.zeros(shape=(len(self.veh_positions),len(self.uav_positions)))
        # for i in range(len(self.veh_positions)):
        #     for j in range(len(self.uav_positions)):
        #         self.PathLoss[i][j] = self.get_path_loss(self.veh_positions[i], self.uav_positions[j])
                
    def update_shadow(self, veh_delta_distance_list, uav_delta_distance_list):
        if len(self.Shadow) != len(veh_delta_distance_list):
            # 1 如果过去一个时间片的车辆数量发生了变化（只会增加）
            Shadow = np.random.normal(0, self.shadow_std, size=(self.n_Veh, self.n_UAV))
            self.Shadow = Shadow
        delta_distance = np.zeros((len(veh_delta_distance_list), len(uav_delta_distance_list)))
        for i in range(len(delta_distance)):
            for j in range(len(delta_distance[0])):
                delta_distance[i][j] = veh_delta_distance_list[i] + uav_delta_distance_list[j]
        if len(veh_delta_distance_list) != 0 or len(uav_delta_distance_list) != 0: 
            self.Shadow = 10 * np.log10(np.exp(-1*(delta_distance/self.decorrelation_distance))* (10 ** (self.Shadow / 10)) + np.sqrt(1-np.exp(-2*(delta_distance/self.decorrelation_distance)))*(10**(np.random.normal(0,self.shadow_std, size=(self.n_Veh, self.n_UAV))/10)))

    def update_fast_fading(self):
        '''快衰落，网上开源代码'''
        gaussian1 = np.random.normal(size=(self.n_Veh, self.n_UAV, self.n_RB))
        gaussian2 = np.random.normal(size=(self.n_Veh, self.n_UAV, self.n_RB))
        # 计算瑞利分布的信道增益
        r = np.sqrt(gaussian1 ** 2 + gaussian2 ** 2)
        # 计算信道增益的平均值
        omega = np.mean(r ** 2)
        # 计算瑞利分布的概率密度函数
        p_r = (2 * r / omega) * np.exp(-r ** 2 / omega)
        # 转换为 dB 单位
        self.FastFading = 10 * np.log10(p_r)

    @staticmethod
    def calculate_path_loss(distance, h_bs=100, fc=2):
        # distance: the distance between UAV and vehicle
        # h_bs: the height of UAV
        # fc: the frequency of UAV
        
        def PL_Los(d):
            return 30.9 + (22.25 - 0.5 * np.log10(h_bs)) * np.log10(d) + 20 * np.log10(fc)
        
        def PL_NLos(d1, d2):
            return np.maximum(PL_Los(d1), 32.4 + (43.2 - 7.6 * np.log10(h_bs)) * np.log10(d2) + 20 * np.log10(fc))
        
        D_H = np.sqrt(distance**2 + h_bs**2)
        d_0 = np.maximum((294.05 * np.log10(h_bs) - 432.94), 18)
        p_1 = 233.98 * np.log10(h_bs) - 0.95

        P_Los = np.where(D_H <= d_0, 1.0, d_0 / D_H + np.exp(-(D_H / p_1) * (1 - (d_0 / D_H))))
        P_Los = np.clip(P_Los, 0, 1)

        P_NLos = 1 - P_Los
        PL = P_Los * PL_Los(np.hypot(distance, h_bs)) + P_NLos * np.minimum(PL_NLos(h_bs, distance), PL_NLos(distance, h_bs))

        return PL


    def get_path_loss_matrix(self, veh_positions, uav_positions):
        veh_positions = np.array(veh_positions)
        uav_positions = np.array(uav_positions)
        
        d1_matrix = np.abs(np.repeat(veh_positions[:, np.newaxis, 0], self.n_UAV, axis=1) - uav_positions[:, 0])
        d2_matrix = np.abs(np.repeat(veh_positions[:, np.newaxis, 1], self.n_UAV, axis=1) - uav_positions[:, 1])
        d_matrix = np.hypot(d1_matrix, d2_matrix) + 0.001
        
        def PL_Los(d):
            return 30.9 + (22.25 - 0.5 * np.log10(self.h_bs)) * np.log10(d) + 20 * np.log10(self.fc)
        
        def PL_NLos(d1, d2):
            return np.maximum(PL_Los(d1), 32.4 + (43.2 - 7.6 * np.log10(self.h_bs)) * np.log10(d2) + 20 * np.log10(self.fc))
        
        D_H = np.sqrt(np.square(d_matrix) + np.square(self.h_bs))
        d_0 = np.maximum((294.05 * np.log10(self.h_bs) - 432.94), 18)
        p_1 = 233.98 * np.log10(self.h_bs) - 0.95
        
        P_Los = np.where(D_H <= d_0, 1.0, d_0 / D_H + np.exp(-(D_H / p_1) * (1 - (d_0 / D_H))))
        P_Los = np.clip(P_Los, 0, 1)
        
        P_NLos = 1 - P_Los
        PL_matrix = P_Los * PL_Los(np.hypot(d_matrix, self.h_bs)) + P_NLos * np.minimum(PL_NLos(self.h_bs, d_matrix), PL_NLos(d_matrix, self.h_bs))
        
        return PL_matrix

    def get_path_loss(self, position_A, position_B):
        '''出自3GPP Release 15的LoS和NLoS模型'''
        # R. Zhong, X. Liu, Y. Liu and Y. Chen, "Multi-Agent Reinforcement Learning in NOMA-aided UAV Networks for Cellular Offloading,"in IEEE Transactions on Wireless Communications, doi: 10.1109/TWC.2021.3104633.
        d1 = abs(position_A[0] - position_B[0])
        d2 = abs(position_A[1] - position_B[1])
        d = math.hypot(d1,d2) + 0.001
        def PL_Los(d):
            return 30.9 + (22.25-0.5*math.log(self.h_bs,10))*math.log10(d) + 20*math.log10(self.fc)
        def PL_NLos(d1, d2):
            return np.max([PL_Los(d1), 32.4+(43.2-7.6*math.log10(self.h_bs))*math.log10(d2)+20*math.log10(self.fc)])
            
        D_H = np.sqrt(np.square(d)+np.square(self.h_bs)) # calculate distance
        d_0 = np.max([(294.05*math.log(self.h_bs,10)-432.94),18])
        p_1 = 233.98*math.log(self.h_bs,10) - 0.95
        if D_H <= d_0:
            P_Los = 1.0
        else:
            P_Los = d_0/D_H + math.exp(-(D_H/p_1)*(1-(d_0/D_H)))

        if P_Los>1:
            P_Los = 1

        P_NLos = 1 - P_Los
        PL = P_Los * PL_Los(math.hypot(d,self.h_bs)) + P_NLos * min(PL_NLos(self.h_bs, d),PL_NLos(d,self.h_bs))
        return PL
class U2IChannel: 
    # U2I仿真信道
    def __init__(self, n_RB, n_BS, n_UAV, hei_UAV, BS_positions):
        '''V2I只存在于一个BS范围内的V2I channel'''
        self.h_bs = 25 # 基站高度25m
        self.h_ms = hei_UAV
        self.hei_UAV = hei_UAV
        self.Decorrelation_distance = 50 
        self.shadow_std = 8
        self.n_BS = n_BS
        self.n_UAV = n_UAV
        self.n_RB = n_RB
        self.fc = 2
        self.BS_positions = BS_positions
        self.update_shadow([])

    def update_positions(self, UAV_positions):
        self.UAV_positions = UAV_positions
        
    def update_pathloss(self):
        self.PathLoss = np.zeros(shape=(len(self.UAV_positions),len(self.BS_positions)))
        for i in range(len(self.UAV_positions)):
            for j in range(len(self.BS_positions)):
                self.PathLoss[i][j] = self.get_path_loss(self.UAV_positions[i], self.BS_positions[j])

    def update_shadow(self, delta_distance_list):
        if len(delta_distance_list) == 0:  # initialization
            self.Shadow = np.random.normal(0, self.shadow_std, size=(self.n_UAV, self.n_BS))
        else: 
            delta_distance = np.repeat(delta_distance_list[:,np.newaxis], self.n_BS, axis=1)
            self.Shadow = 10 * np.log10(np.exp(-1*(delta_distance/self.Decorrelation_distance))* (10 ** (self.Shadow / 10)) + np.sqrt(1-np.exp(-2*(delta_distance/self.Decorrelation_distance)))*(10**(np.random.normal(0,self.shadow_std, size=(self.n_UAV, self.n_BS))/10)))

    def update_fast_fading(self):
        gaussian1 = np.random.normal(size=(self.n_UAV, self.n_BS, self.n_RB))
        gaussian2 = np.random.normal(size=(self.n_UAV, self.n_BS, self.n_RB))
        # 计算瑞利分布的信道增益
        r = np.sqrt(gaussian1 ** 2 + gaussian2 ** 2)
        # 计算信道增益的平均值
        omega = np.mean(r ** 2)
        # 计算瑞利分布的概率密度函数
        p_r = (2 * r / omega) * np.exp(-r ** 2 / omega)
        # 转换为 dB 单位
        self.FastFading = 10 * np.log10(p_r)

    
    def get_path_loss(self, position_A, position_B):
        d1 = abs(position_A[0] - position_B[0])
        d2 = abs(position_A[1] - position_B[1])
        d_2 = math.hypot(d1,d2)+0.001 
        if d_2 > 4000:
            return 1000 # 只允许2d距离在4000米以内的传输，超过了就置为极小值
        d_3 = math.hypot(d_2, self.hei_UAV) + 0.001
        # 计算LoS概率
        log_hu = math.log10(self.hei_UAV)
        p1 = 4300 * log_hu - 3800
        d1 = max(460 * log_hu - 700, 18)
        poss_los = 0
        if d_2 <= d1:
            poss_los = 1
        else:
            poss_los = d1/d_2 + math.exp(-d_2 / p1) * (1-d1/d_2)
        if self.hei_UAV > 100:
            poss_los = 1
        def PL_NLOS(d):
            return -17.5 + (46-7*log_hu) * math.log10(d) + 20 * math.log10(40 * math.pi * self.fc / 3)
        def PL_LOS(d):
            return 28.0 + 22 * math.log10(d ) + 20 * math.log10(self.fc)
        
        return poss_los * PL_LOS(d_3) + (1-poss_los) * PL_NLOS(d_3)
class U2UChannel:
    def __init__(self, n_RB, n_UAV, hei_UAV):
        # M. M. Azari, G. Geraci, A. Garcia-Rodriguez and S. Pollin, "UAV-to-UAV Communications in Cellular Networks," in IEEE Transactions on Wireless Communications, 2020, doi: 10.1109/TWC.2020.3000303.
        # A Survey of Channel Modeling for UAV Communications
        self.t = 0
        self.h_bs = hei_UAV # 车作为BS的高度 
        self.h_ms = hei_UAV # 车作为MS的高度
        self.fc = 2 # carrier frequency
        self.decorrelation_distance = 50
        self.shadow_std = 3 # shadow的标准值
        self.n_UAV = n_UAV
        self.n_RB = n_RB # RB数量
        self.update_shadow([])

    def update_positions(self, uav_positions):
        '''更新无人机的位置'''
        self.positions = uav_positions

    def update_pathloss(self):
        self.PathLoss = np.zeros(shape=(len(self.positions),len(self.positions)))
        for i in range(len(self.positions)):
            for j in range(len(self.positions)):
                if i == j: continue
                self.PathLoss[i][j] = self.get_path_loss(self.positions[i], self.positions[j])
                
    def update_shadow(self, delta_distance_list):
        '''输入距离变化，计算阴影变化，基于3GPP的规范'''
        delta_distance = np.zeros((len(delta_distance_list), len(delta_distance_list)))
        for i in range(len(delta_distance)):
            for j in range(len(delta_distance)):
                delta_distance[i][j] = delta_distance_list[i] + delta_distance_list[j]
        if len(delta_distance_list) == 0: 
            self.Shadow = np.random.normal(0,self.shadow_std, size=(self.n_UAV, self.n_UAV))
        else:
            self.Shadow = 10 * np.log10(np.exp(-1*(delta_distance/self.decorrelation_distance))* (10 ** (self.Shadow / 10)) + np.sqrt(1-np.exp(-2*(delta_distance/self.decorrelation_distance)))*(10**(np.random.normal(0,self.shadow_std, size=(self.n_UAV, self.n_UAV))/10)))
        np.fill_diagonal(self.Shadow, 0)

    def update_fast_fading(self):
        '''快衰落，网上开源代码'''
        gaussian1 = np.random.normal(size=(self.n_UAV, self.n_UAV, self.n_RB))
        gaussian2 = np.random.normal(size=(self.n_UAV, self.n_UAV, self.n_RB))
        # 计算瑞利分布的信道增益
        r = np.sqrt(gaussian1 ** 2 + gaussian2 ** 2)
        # 计算信道增益的平均值
        omega = np.mean(r ** 2)
        # 计算瑞利分布的概率密度函数
        p_r = (2 * r / omega) * np.exp(-r ** 2 / omega)
        # 转换为 dB 单位
        self.FastFading = 10 * np.log10(p_r)
        for i in range(self.n_RB):
            np.fill_diagonal(self.FastFading[:, :, i], 0)

    def get_path_loss(self, position_A, position_B):
        '''U2U path loss'''
        # A Survey of Channel Modeling for UAV Communications
        d1 = abs(position_A[0] - position_B[0])
        d2 = abs(position_A[1] - position_B[1])
        d = math.hypot(d1,d2)+0.001 
        alpha_uu = 2.05
        PL = 10 * alpha_uu * math.log10(d)
        return PL
class I2IChannel:
    def __init__(self, n_RB, n_BS, BS_positions):
        self.t = 0
        self.h_bs = 25 # 车作为BS的高度 
        self.h_ms = 25 # 车作为MS的高度
        self.fc = 2 # carrier frequency
        self.decorrelation_distance = 50
        self.shadow_std = 3 # shadow的标准值
        self.n_BS = n_BS # 车辆数量
        self.n_RB = n_RB # RB数量
        self.positions = BS_positions
        self.update_shadow()

    def update_pathloss(self):
        self.PathLoss = np.zeros(shape=(len(self.positions),len(self.positions)))
        for i in range(len(self.positions)):
            for j in range(len(self.positions)):
                self.PathLoss[i][j] = self.get_path_loss(self.positions[i], self.positions[j])
                
    def update_shadow(self):
        self.Shadow = np.random.normal(0,self.shadow_std, size=(self.n_BS, self.n_BS))

    def get_path_loss(self, position_A, position_B):
        d1 = abs(position_A[0] - position_B[0])
        d2 = abs(position_A[1] - position_B[1])
        d = math.hypot(d1,d2)+0.001
        d_bp = 4 * (self.h_bs - 1) * (self.h_ms - 1) * self.fc * (10**9)/(3*10**8)     
        def PL_Los(d):
            if d < d_bp:
                return 22.7 * np.log10(d) + 41 + 20 * np.log10(self.fc/5)
            else:
                return 40.0 * np.log10(d) + 9.45 - 17.3 * np.log10(self.h_bs) - 17.3 * np.log10(self.h_ms) + 2.7 * np.log10(self.fc/5)
        def PL_NLos(d_a,d_b):
                n_j = max(2.8 - 0.0024*d_b, 1.84)
                return PL_Los(d_a) + 20 - 12.5*n_j + 10 * n_j * np.log10(d_b) + 3*np.log10(self.fc/5)
        if min(d1,d2) < 100: # 以100m作为LoS存在的标准
            PL = PL_Los(d)
            self.ifLOS = True
            self.shadow_std = 3
        else:
            PL = min(PL_NLos(d1,d2), PL_NLos(d2,d1))
            self.ifLOS = False
            self.shadow_std = 4 
        return PL