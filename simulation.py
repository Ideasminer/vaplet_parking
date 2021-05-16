import numpy as np
import time

class Layout:
    def __init__(self, num_isl, k, num_tpark, num_pass = 1, num_stack=10, spot_size = (2.5, 5), clear_way = 3):
        # num_isl	- number of parking island (int)
        # k			- the value of k in each k-stack. Every island contain 2k columns. (list, size = isl_num)
        # num_tpark	- the number of temporary parking lane in each gap. (list, size = isl_num + 1)
        # num_pass 	- the number of passing lane in each gap. (list, size = isl_num + 1, default 1, means all gaps contain only 1 passing lane)
        # num_stack	- the number of k-stack in each island. (int, default 10)
        # spot_size	- the size of parking spot. (tuple, default (2, 5))
        # clear_way	- way for getting into / out the facility. (int, only need width, defalut 3)
        isl_info = []
        dwell_info = []
        arrive_info = []
        for isl_ind in range(num_isl):
            isl_info.append(np.zeros([num_stack, 2 * k[isl_ind]]))
            dwell_info.append(np.zeros([num_stack, 2 * k[isl_ind]]))
            arrive_info.append(np.zeros([num_stack, 2 * k[isl_ind]]))
        tpark_info = []
        tpark_capacity = int(np.floor(num_stack * spot_size[0] / spot_size[1]))
        if len(num_tpark) != num_isl + 1:
            raise ValueError("Please input the right num_tpark")
        for tpark_ind in range(len(num_tpark)):
            tpark_info.append(np.zeros([tpark_capacity, int(num_tpark[tpark_ind])]))
        pass_info = []
        pass_capacity = int(np.floor(num_stack * spot_size[0] / spot_size[1]))
        if isinstance(num_pass, int):
            num_pass = [num_pass for i in range(num_isl + 1)]
        if len(num_pass) != num_isl + 1:
            raise ValueError("Please input the right num_pass")
        for pass_ind in range(len(num_pass)):
            pass_info.append(np.zeros([pass_capacity,num_pass[pass_ind]]))
        self.isl_info = isl_info
        self.dwell_info = dwell_info
        self.tpark_info = tpark_info
        self.pass_info = pass_info
        self.spot_size = spot_size
       	self.clear_way = clear_way
        self.arrive_info = arrive_info


    def get_size(self):
        # get the parking facility's overall size
        tpark_x = self.spot_size[0] * sum([tpark.shape[1] for tpark in self.tpark_info])
        pass_x = self.spot_size[0] * sum([pass_lane.shape[1] for pass_lane in self.pass_info])
        island_x = self.spot_size[1] * sum([island.shape[1] for island in self.isl_info])
        try:
            whole_x = tpark_x + pass_x + island_x
            whole_y = self.clear_way + self.isl_info[0].shape[0] * self.spot_size[0]
            return whole_x, whole_y
        except:
            print(self.isl_info)
            raise ValueError("list index out of range")


    def get_capacity(self):
        return sum(island.shape[0] * island.shape[1] for island in self.isl_info)


def generate(lam, mu, v_num, threshold = 4):
    # lam 	- arrival rate
    # mu	- mean of dwell time
    # v_num	- number of veh which the facility can hold
    arrival = np.cumsum(np.random.exponential(lam, v_num))
    dwell = np.random.exponential(mu, v_num)
    dwell_type = np.array([1 if i <= threshold else -1 for i in dwell])
    departure = arrival + dwell
    event = np.append(arrival, departure)
    ind = np.argsort(event)
    ind = [i + 1 for i in ind]
    event = np.sort(event)
    demand = v_num
    return event, ind, demand, dwell_type



def arrive(veh, turn, facility, dwell, event, policy = "A1", show = False, mu=4):
    # 在选择到达策略前，首先判断该到达车辆是否是tpark中重定位的车辆
    for i in range(len(facility.tpark_info)):
        if np.argwhere(facility.tpark_info[i] == veh).size > 0:
            # 如果在tpark中找到了该车，由于车辆排列，不可能出现block，因此直接清空车位进场。
            veh_arg = np.argwhere(facility.tpark_info[i] == veh)[0]
            facility.tpark_info[i][veh_arg[0]][veh_arg[1]] = 0
            break
    if policy == "A1":
        # 从最小的岛开始停车，从中间开始停放，停满了后，停在次小的岛
        # 第一步，选在还有车位的最小的岛
        isl_capacity = []
        backup_isl = []
        if np.argwhere(facility.isl_info == veh).size > 0:
            raise ValueError("Already Here!!!!\n\n\n")
        for i in range(len(facility.isl_info)):
            island = facility.isl_info[i]
            if np.argwhere(island == 0).size > 0:
                # 如果当前停车岛还有空车位，成为备选停车岛
                backup_isl.append(i)
                isl_capacity.append(island.shape[0] * island.shape[1])
            else:
                continue
        if backup_isl:
            # 如果还有备选停车岛，说明停车场没有停满
            # 选择还有车位的最小停车岛
            target_isl_ind = backup_isl[np.argmin(isl_capacity)]
            target_isl = facility.isl_info[target_isl_ind]
            # 检测其中间列是否还有空位
            for i in range(target_isl.shape[1] // 2):
                left_row = int(target_isl.shape[1] / 2 - i - 1)
                right_row = int(target_isl.shape[1] / 2 + i)
                # 首先检测左侧是否有空位，如果有，停放在左侧
                if np.argwhere(target_isl[ : ,left_row] == 0).size > 0:
                    empty_ind = np.argwhere(target_isl[ : ,left_row] == 0)[0][0]
                    target_isl[empty_ind][left_row] = veh
                    break
                # 如果左侧没空，右侧有空，则停放在右侧
                elif np.argwhere(target_isl[ : ,right_row] == 0).size > 0:
                    empty_ind = np.argwhere(target_isl[ : ,right_row] == 0)[0][0]
                    target_isl[empty_ind][right_row] = veh
                    break
                else:
                    # 如果当前列都没空，则看外侧一列是否有空
                    continue
            facility.isl_info[target_isl_ind] = target_isl
            return "accept"
        else:
            return "reject"
    if policy == "A2":
        # 先停放在中间排，所有停车岛的中间排停放慢了，再停放在外侧排
        isl_capacity = []
        backup_isl = []
        for i in range(len(facility.isl_info)):
            island = facility.isl_info[i]
            if np.argwhere(island == 0).size > 0:
                # 如果当前停车岛还有空车位，成为备选停车岛
                backup_isl.append(i)
                isl_capacity.append(island.shape[0] * island.shape[1])
            else:
                continue
        if backup_isl:
            # 如果还有备选停车岛，说明停车场没有停满
            # 在备选停车岛中检测是否有空车位的中间排
            # 从最小的停车岛开始检测，如果当前停车岛的第i排没有车位，则检测下一个停车岛，如果都没有，检测外侧排
            # 一个数组储存各个停车岛的排数
            isl_capacity_argsort = np.argsort(np.array(isl_capacity))
            backup_isl = [backup_isl[i] for i in isl_capacity_argsort]
            island_k = [facility.isl_info[i].shape[1] // 2 for i in backup_isl]
            is_parked = 0
            for k in range(max(island_k)):
                if is_parked == 1:
                    break
                for i in range(len(backup_isl)):
                    isl_ind = backup_isl[i]
                    island = facility.isl_info[isl_ind]
                    this_k = island_k[i]
                    if this_k < k:
                        # 如果当前停车岛的排数小于当前循环的排数，说明当前停车岛已经检测完，直接到下一个停车岛
                        continue
                    left_row = this_k - k - 1
                    right_row = this_k + k
                    # 首先检测左侧是否有空位，如果有，停放在左侧
                    if np.argwhere(island[ : ,left_row] == 0).size > 0:
                        empty_ind = np.argwhere(island[ : ,left_row] == 0)[0][0]
                        island[empty_ind][left_row] = veh
                        is_parked = 1
                        break
                    # 如果左侧没有空位，检测右侧是否有空位
                    elif np.argwhere(island[ : ,right_row] == 0).size > 0:
                        empty_ind = np.argwhere(island[ : ,right_row] == 0)[0][0]
                        island[empty_ind][right_row] = veh
                        is_parked = 1
                        break
                    # 如果都没有空位，则检测下一个停车岛的该排
                    else:
                        continue
            return "accept"
        else:
            return "reject"
    if policy == "C1":
        # 分长时间停车和短时间停车，短时间停车在左侧停放，长时间停车在右侧停放，从中间开始停放
        # facility中不光需要记录停放的车辆标号，还要存放其类型，分别用1和-1表示。0表示空缺
        # 首先，判断是否已经满了（是否有备选停车岛），满了的话，直接拒绝
        backup_isl_l = []
        isl_capacity_l = []
        backup_isl_r = []
        isl_capacity_r = []
        for i in range(len(facility.isl_info)):
            if np.argwhere(facility.isl_info[i][:, : facility.isl_info[i].shape[1] // 2] == 0).size > 0:
                # 如果当前停车岛左侧有空车位，则成为短时备选停车岛
                backup_isl_l.append(i)
                isl_capacity_l.append(facility.isl_info[i].shape[0] * facility.isl_info[i].shape[1])
            if np.argwhere(facility.isl_info[i][:, facility.isl_info[i].shape[1] // 2 : ] == 0).size > 0:
                # 如果当前停车岛右侧有空车位，则成为长时备选停车岛
                backup_isl_r.append(i)
                isl_capacity_r.append(facility.isl_info[i].shape[0] * facility.isl_info[i].shape[1])
        if dwell == 1:
            if backup_isl_l:
                # 如果有备选的短时停车岛，选择其中最小的，当作目前的停车岛
                target_isl_ind = backup_isl_l[np.argmin(isl_capacity_l)]
                target_isl = facility.isl_info[target_isl_ind]
                target_dwell = facility.dwell_info[target_isl_ind]
                # 循环，从中间开始停放，停放在左侧
                for i in range(target_isl.shape[1] // 2):
                    left_row = int(target_isl.shape[1] / 2 - i - 1)
                    # 判断是否左侧有空
                    if np.argwhere(target_isl[ : ,left_row] == 0).size > 0:
                        empty_ind = np.argwhere(target_isl[ : ,left_row] == 0)[0][0]
                        target_isl[empty_ind][left_row] = veh
                        # 记录停车时长类型
                        target_dwell[empty_ind][left_row] = dwell
                        # 如果有空，停放后，直接跳出循环
                        break
                    # 如果没空，则直接到下一个循环（即下一排）
                    else:
                        continue
                facility.isl_info[target_isl_ind] = target_isl
                facility.dwell_info[target_isl_ind] = target_dwell
                return "accept"
            else:
                return "reject"
        elif dwell == -1:
            if backup_isl_r:
            # 如果有备选的短时停车岛，选择其中最小的，当作目前的停车岛
                target_isl_ind = backup_isl_r[np.argmin(isl_capacity_r)]
                target_isl = facility.isl_info[target_isl_ind]
                target_dwell = facility.dwell_info[target_isl_ind]
                for i in range(target_isl.shape[1] // 2):
                    right_row = int(target_isl.shape[1] / 2 + i)
                    # 判断右侧是否有空
                    if np.argwhere(target_isl[ : ,right_row] == 0).size > 0:
                        empty_ind = np.argwhere(target_isl[ : ,right_row] == 0)[0][0]
                        # 停放
                        target_isl[empty_ind][right_row] = veh
                        # 记录停车时长类型
                        target_dwell[empty_ind][right_row] = dwell
                        # 如果有空，停放后，直接跳出循环
                        break
                    else:
                        # 如果没空，则直接到下一个循环（即下一排）
                        continue
                facility.isl_info[target_isl_ind] = target_isl
                facility.dwell_info[target_isl_ind] = target_dwell
                return "accept"
            else:
                return "reject"
        else:
            # 如果类型不是1不是-1，则说明生成出错，报错退出
            raise ValueError("Wrong Dwell Type Detected")
    if policy == "C2":
        # 左侧停放短时间，右侧停放长时间，从中间开始停放。
        # 长时停车从最大的岛开始停放，停满为止
        # 短时间停车从最小的岛开始停放，停满位为止
        backup_isl_l = []
        isl_capacity_l = []
        backup_isl_r = []
        isl_capacity_r = []
        for i in range(len(facility.isl_info)):
            if np.argwhere(facility.isl_info[i][:, : facility.isl_info[i].shape[1] // 2] == 0).size > 0:
                # 如果当前停车岛左侧有空车位，则成为短时备选停车岛
                backup_isl_l.append(i)
                isl_capacity_l.append(facility.isl_info[i].shape[0] * facility.isl_info[i].shape[1])
            if np.argwhere(facility.isl_info[i][:, facility.isl_info[i].shape[1] // 2 : ] == 0).size > 0:
                # 如果当前停车岛右侧有空车位，则成为长时备选停车岛
                backup_isl_r.append(i)
                isl_capacity_r.append(facility.isl_info[i].shape[0] * facility.isl_info[i].shape[1])
                # 当前停车岛左右都满了，继续寻找下一个备选停车岛
        if dwell == 1:
            if backup_isl_l:
                # 如果有备选的短时停车岛，选择其中最小的，当作目前的停车岛
                target_isl_ind = backup_isl_l[np.argmin(isl_capacity_l)]
                target_isl = facility.isl_info[target_isl_ind]
                target_dwell = facility.dwell_info[target_isl_ind]
                # 循环，从中间开始停放，停放在左侧
                for i in range(target_isl.shape[1] // 2):
                    left_row = int(target_isl.shape[1] / 2 - i - 1)
                    # 判断是否左侧有空
                    if np.argwhere(target_isl[ : ,left_row] == 0).size > 0:
                        empty_ind = np.argwhere(target_isl[ : ,left_row] == 0)[0][0]
                        target_isl[empty_ind][left_row] = veh
                        # 记录停车时长类型
                        target_dwell[empty_ind][left_row] = dwell
                        # 如果有空，停放后，直接跳出循环
                        break
                    # 如果没空，则直接到下一个循环（即下一排）
                    else:
                        continue
                facility.isl_info[target_isl_ind] = target_isl
                facility.dwell_info[target_isl_ind] = target_dwell
                return "accept"
            else:
                return "reject"
        elif dwell == -1:
            if backup_isl_r:
            # 如果有备选的长时停车岛，选择其中最大的，当作目前的停车岛
                target_isl_ind = backup_isl_r[np.argmax(isl_capacity_r)]
                target_isl = facility.isl_info[target_isl_ind]
                target_dwell = facility.dwell_info[target_isl_ind]
                for i in range(target_isl.shape[1] // 2):
                    right_row = int(target_isl.shape[1] / 2 + i)
                    # 判断右侧是否有空
                    if np.argwhere(target_isl[ : ,right_row] == 0).size > 0:
                        empty_ind = np.argwhere(target_isl[ : ,right_row] == 0)[0][0]
                        # 停放
                        target_isl[empty_ind][right_row] = veh
                        # 记录停车时长类型
                        target_dwell[empty_ind][right_row] = dwell
                        # 如果有空，停放后，直接跳出循环
                        break
                    else:
                        # 如果没空，则直接到下一个循环（即下一排）
                        continue
                facility.isl_info[target_isl_ind] = target_isl
                facility.dwell_info[target_isl_ind] = target_dwell
                return "accept"
            else:
                return "reject"
        else:
            # 如果类型不是1不是-1，则说明生成出错，报错退出
            raise ValueError("Wrong Dwell Type Detected")
    if policy == "C3":
        # 从最小的岛开始停放短时
        # 从最大的岛开始停放长时
        # 不分左右，停满为止
        # 首先检测是否还有候选停车岛
        backup_isl = []
        isl_capacity = []
        for i in range(len(facility.isl_info)):
            island = facility.isl_info[i]
            if np.argwhere(island == 0).size > 0:
                # 如果有还有空车位的停车岛
                backup_isl.append(i)
                isl_capacity.append(island.shape[0] * island.shape[1])
        # 如果还有备选停车岛
        if backup_isl:
            # 将停车岛按容量从小到大排序
            isl_capacity_argsort = np.argsort(isl_capacity)
            # 如果此时是短时类型
            if dwell == 1:
                # 如果是短时停放，目标停车岛是当前候选停车岛中最小的停车岛
                target_isl_ind = backup_isl[isl_capacity_argsort[0]]
                target_isl = facility.isl_info[target_isl_ind]
                target_dwell = facility.dwell_info[target_isl_ind]
            elif dwell == -1:
                # 如果是长时停放，目标停车岛是当前候选停车岛中最大的停车岛
                target_isl_ind = backup_isl[isl_capacity_argsort[-1]]
                target_isl = facility.isl_info[target_isl_ind]
                target_dwell = facility.dwell_info[target_isl_ind]
            # 然后从中间列开始检测，哪里有空停哪里
            for i in range(target_isl.shape[1] // 2):
                left_row = int(target_isl.shape[1] // 2 - i - 1)
                right_row = int(target_isl.shape[1] // 2 + i)
                if np.argwhere(target_isl[: , left_row] == 0).size > 0:
                    empty_ind = np.argwhere(target_isl[: , left_row] == 0)[0][0]
                    target_isl[empty_ind][left_row] = veh
                    target_dwell[empty_ind][left_row] = dwell
                    break
                elif np.argwhere(target_isl[: , right_row] == 0).size > 0:
                    empty_ind = np.argwhere(target_isl[: , right_row] == 0)[0][0]
                    target_isl[empty_ind][right_row] = veh
                    target_dwell[empty_ind][left_row] = dwell
                    break
                else:
                    # 当前左右都没空，继续循环，找下一排
                    continue
            return "accept"
        else:
            # 如果没有备选停车岛了，拒绝入场
            # print(veh, facility.isl_info)
            return "reject"
    if policy == "C4":
        # 从中央排开始停放
        # 但是如果前一排有长期停车，后面来了一个短时停车，则短时停车会停放在该长时停车车辆的后方
        # 首先，先获取备选停车岛
        backup_isl = []
        isl_capacity = []
        for i in range(len(facility.isl_info)):
            island = facility.isl_info[i]
            if np.argwhere(island == 0).size > 0:
                # 如果有还有空车位的停车岛
                backup_isl.append(i)
                isl_capacity.append(island.shape[0] * island.shape[1])
        # 如果还有备选停车岛
        if backup_isl:
            # 从最小的停车岛开始停放
            target_isl_ind = backup_isl[np.argmin(isl_capacity)]
            target_isl = facility.isl_info[target_isl_ind]
            target_dwell = facility.dwell_info[target_isl_ind]
            # 从中间排开始停放
            for i in range(target_isl.shape[1] // 2):
                left_row = target_isl.shape[1] // 2 - i - 1
                right_row = target_isl.shape[1] // 2 + i
                # 如果当前排是中间排，不分长短，直接停放
                if i == 0:
                    if np.argwhere(target_isl[:,left_row] == 0).size > 0:
                        empty_ind = np.argwhere(target_isl[:,left_row] == 0)[0][0]
                        target_isl[empty_ind][left_row] = veh
                        target_dwell[empty_ind][left_row] = dwell
                        return "accept"
                    elif np.argwhere(target_isl[:,right_row] == 0).size > 0:
                        empty_ind = np.argwhere(target_isl[:,right_row] == 0)[0][0]
                        target_isl[empty_ind][right_row] = veh
                        target_dwell[empty_ind][right_row] = dwell
                        return "accept"
                    else:
                        continue
                else:
                    # 如果当前排不是中间排，则需要检索其前面一排是否有长时停车
                    if np.argwhere(target_isl[:,left_row] == 0).size > 0:
                        if np.argwhere(target_dwell[:,left_row + 1] == -1).size > 0:
                            # 如果当前排有空位且前一排有长时停车的
                            empty_ind = np.argwhere(target_dwell[:,left_row + 1] == -1)
                            c4count = 0
                            for e in empty_ind:
                                e = e[0]
                                if target_isl[e][left_row] == 0:
                                    # 如果长时停车后方有空位
                                    target_isl[e][left_row] = veh
                                    target_dwell[e][left_row] = dwell
                                    return "accept"
                                else:
                                    c4count += 1
                            if c4count >= len(empty_ind):
                                # 如果所有的长时停车后方均没有空位
                                # 则随便停在当前排的一个空位
                                empty_ind = np.argwhere(target_isl[:,left_row] == 0)[0][0]
                                target_isl[empty_ind][left_row] = veh
                                target_dwell[empty_ind][left_row] = dwell
                                return "accept"
                        else:
                            # 如果当前排有空位，但是前面一排没有长时停车的
                            # 则随便停放算了
                            empty_ind = np.argwhere(target_isl[:,left_row] == 0)[0][0]
                            target_isl[empty_ind][left_row] = veh
                            target_dwell[empty_ind][left_row] = dwell
                            return "accept"
                    elif np.argwhere(target_isl[:,right_row] == 0).size > 0:
                        if np.argwhere(target_dwell[:,right_row - 1] == -1).size > 0:
                            # 如果当前排有空位且前一排有长时停车的
                            empty_ind = np.argwhere(target_dwell[:,right_row - 1] == -1)
                            c4count = 0
                            for e in empty_ind:
                                e = e[0]
                                if target_isl[e][right_row] == 0:
                                    # 如果长时停车后方有空位
                                    target_isl[e][right_row] = veh
                                    target_dwell[e][right_row] = dwell
                                    return "accept"
                                else:
                                    c4count += 1
                            if c4count >= len(empty_ind):
                                # 如果所有的长时停车后方均没有空位
                                # 则随便停在当前排的一个空位
                                empty_ind = np.argwhere(target_isl[:,right_row] == 0)[0][0]
                                target_isl[empty_ind][right_row] = veh
                                target_dwell[empty_ind][right_row] = dwell
                                return "accept"
                        else:
                            # 如果当前排有空位，但是前面一排没有长时停车的
                            # 则随便停放算了
                            empty_ind = np.argwhere(target_isl[:,right_row] == 0)[0][0]
                            target_isl[empty_ind][right_row] = veh
                            target_dwell[empty_ind][right_row] = dwell
                            return "accept"
                    else:
                        # 如果当前排没有空位
                        continue
            return "accept"
        else:
            return "reject"
    if policy == "C5":
        # 左侧停放短时，右侧停放长时
        # 先停放在中间排，所有停车岛的中间排停放慢了再停放在外侧排
        # 短时从最小的岛开始停放，长时从最大的岛开始停放
        isl_capacity = []
        backup_isl = []
        for i in range(len(facility.isl_info)):
            island = facility.isl_info[i]
            if np.argwhere(island == 0).size > 0:
                # 如果当前停车岛还有空车位，成为备选停车岛
                backup_isl.append(i)
                isl_capacity.append(island.shape[0] * island.shape[1])
            else:
                continue
        if backup_isl:
            # 如果还有备选停车岛，说明停车场没有停满
            isl_capacity_argsort = np.argsort(np.array(isl_capacity))
            # 根据停车岛的大小，给备选停车岛排序
            backup_isl_l = [backup_isl[i] for i in isl_capacity_argsort]
            backup_isl_r = [backup_isl[i] for i in isl_capacity_argsort[::-1]]
            # 得到备选停车岛的排数(K)
            island_k_l = [facility.isl_info[i].shape[1] // 2 for i in backup_isl_l]
            island_k_r = [facility.isl_info[i].shape[1] // 2 for i in backup_isl_r]
            # 如果是短时停放
            if dwell == 1:
                # 已经停放标志
                is_parked = 0
                for k in range(max(island_k_l)):
                    if is_parked == 1:
                        # 如果已经停放了，则直接退出循环
                        break
                    # 如果没有停放，则检索每个岛，停放在左侧
                    for i in range(len(backup_isl_l)):
                        isl_ind = backup_isl_l[i]
                        island = facility.isl_info[isl_ind]
                        island_dwell = facility.dwell_info[isl_ind]
                        this_k = island_k_l[i]
                        if this_k < k:
                            # 如果当前停车岛的排数小于当前循环的排数，说明当前停车岛已经检测完，直接到下一个停车岛
                            continue
                        # 如果当前岛的排数大于等于当前循环的排数，说明当前岛没有检测完，进行检测
                        left_row = this_k - k - 1
                        # 如果左侧当前排有空位
                        if np.argwhere(island[ : ,left_row] == 0).size > 0:
                            empty_ind = np.argwhere(island[ : ,left_row] == 0)[0][0]
                            island[empty_ind][left_row] = veh
                            # 同时记录停放类型
                            island_dwell[empty_ind][left_row] = dwell
                            # 已经停放标志置1
                            is_parked = 1
                            break
                        else:
                            # 如果左侧排没有车位，则直接进行下次循环（检测下一个岛）
                            continue
            # 如果是长时停放
            elif dwell == -1:
                is_parked = 0
                for k in range(max(island_k_r)):
                    if is_parked == 1:
                        # 如果已经停放了，则直接退出循环
                        break
                    # 如果没有停放，则检索每个岛，停放在右侧
                    for i in range(len(backup_isl_r)):
                        isl_ind = backup_isl_r[i]
                        island = facility.isl_info[isl_ind]
                        island_dwell = facility.dwell_info[isl_ind]
                        this_k = island_k_r[i]
                        if this_k < k:
                            # 如果当前停车岛的排数小于当前循环的排数，说明当前停车岛已经检测完，直接到下一个停车岛
                            continue
                        # 如果当前岛的排数大于等于当前循环的排数，说明当前岛没有检测完，进行检测
                        right_row = this_k + k
                        # 如果左侧当前排有空位
                        if np.argwhere(island[ : ,right_row] == 0).size > 0:
                            empty_ind = np.argwhere(island[ : ,right_row] == 0)[0][0]
                            island[empty_ind][right_row] = veh
                            # 同时记录停放类型
                            island_dwell[empty_ind][right_row] = dwell
                            # 已经停放标志置1
                            is_parked = 1
                            break
                        else:
                            # 如果左侧排没有车位，则直接进行下次循环（检测下一个岛）
                            continue
            else:
                raise ValueError("Wrong Dwell Type Detected")
            return "accept"
        else:
            # print(veh, facility.isl_info)
            return "reject"
    if policy == "B1":
        # 按照阻挡概率进行停放，单岛优先策略
        # 首先检索哪些岛可以停放，从中选择一个最小的停车岛
        # 先从最内侧开始停放，最内侧排有空时，不计算阻挡概率
        # 如果内侧排已经满了，则停放在外侧排，选择停车栈堆时，计算各个栈堆对应位置的阻挡概率
        # 计算公式为：1-1/2*e^(-t/mu)，其中，t是两个车到达时间的差值(t2-t1)
        isl_capacity = []
        backup_isl = []
        probn = 0
        if np.argwhere(facility.isl_info == veh).size > 0:
            raise ValueError("Already Here!!!!\n\n\n")
        for i in range(len(facility.isl_info)):
            island = facility.isl_info[i]
            if np.argwhere(island == 0).size > 0:
                # 如果当前停车岛还有空车位，成为备选停车岛
                backup_isl.append(i)
                isl_capacity.append(island.shape[0] * island.shape[1])
            else:
                continue
        if backup_isl:
            # 如果还有备选停车岛，说明停车场没有停满
            # 选择还有车位的最小停车岛
            target_isl_ind = backup_isl[np.argmin(isl_capacity)]
            target_isl = facility.isl_info[target_isl_ind]
            target_arrive = facility.arrive_info[target_isl_ind]
            # 检测其中间列是否还有空位
            for i in range(target_isl.shape[1] // 2):
                left_row = int(target_isl.shape[1] / 2 - i - 1)
                right_row = int(target_isl.shape[1] / 2 + i)
                # 首先检测左侧是否有空位，如果有，停放在左侧
                if np.argwhere(target_isl[ : ,left_row] == 0).size > 0:
                    if i == 0:
                        probn = 0
                        empty_ind = np.argwhere(target_isl[ : ,left_row] == 0)[0][0]
                    else:
                        probn_ls = [1 - (1/2 * ((np.e)**(- (event - target_arrive[ : ,left_row][a][0]) / mu))) for a in np.argwhere(target_isl[ : ,left_row] == 0)]
                        probn = min(probn_ls)
                        ind_ls = [a[0] for a in np.argwhere(target_isl[ : ,left_row] == 0)]
                        empty_ind = ind_ls[np.argmin(probn)]
                    target_isl[empty_ind][left_row] = veh
                    target_arrive[empty_ind][left_row] = event
                    break
                # 如果左侧没空，右侧有空，则停放在右侧
                elif np.argwhere(target_isl[ : ,right_row] == 0).size > 0:
                    if i == 0:
                        probn = 0
                        empty_ind = np.argwhere(target_isl[ : ,right_row] == 0)[0][0]
                    else:
                        probn_ls = [1 - (1/2 * (np.e**(- (event - target_arrive[ : ,right_row][a][0]) / mu))) for a in np.argwhere(target_isl[ : ,right_row] == 0)]
                        probn = min(probn_ls)
                        ind_ls = [a[0] for a in np.argwhere(target_isl[ : ,right_row] == 0)]
                        empty_ind = ind_ls[np.argmin(probn)]
                    target_isl[empty_ind][right_row] = veh
                    target_arrive[empty_ind][right_row] = event
                    break
                else:
                    # 如果当前列都没空，则看外侧一列是否有空
                    continue
            facility.isl_info[target_isl_ind] = target_isl
            facility.arrive_info[target_isl_ind] = target_arrive
            return "accept"
        else:
            return "reject"
    if policy == "B2":
        isl_capacity = []
        backup_isl = []
        for i in range(len(facility.isl_info)):
            island = facility.isl_info[i]
            if np.argwhere(island == 0).size > 0:
                # 如果当前停车岛还有空车位，成为备选停车岛
                backup_isl.append(i)
                isl_capacity.append(island.shape[0] * island.shape[1])
            else:
                continue
        if backup_isl:
            # 如果还有备选停车岛，说明停车场没有停满
            # 在备选停车岛中检测是否有空车位的中间排
            # 从最小的停车岛开始检测，如果当前停车岛的第i排没有车位，则检测下一个停车岛，如果都没有，检测外侧排
            # 一个数组储存各个停车岛的排数
            isl_capacity_argsort = np.argsort(np.array(isl_capacity))
            backup_isl = [backup_isl[i] for i in isl_capacity_argsort]
            island_k = [facility.isl_info[i].shape[1] // 2 for i in backup_isl]
            is_parked = 0
            probn = 0
            for k in range(max(island_k)):
                if is_parked == 1:
                    break
                for i in range(len(backup_isl)):
                    isl_ind = backup_isl[i]
                    island = facility.isl_info[isl_ind]
                    this_k = island_k[i]
                    isl_arrive = facility.arrive_info[isl_ind]
                    if this_k < k:
                        # 如果当前停车岛的排数小于当前循环的排数，说明当前停车岛已经检测完，直接到下一个停车岛
                        continue
                    left_row = this_k - k - 1
                    right_row = this_k + k
                    # 首先检测左侧是否有空位，如果有，停放在左侧
                    if np.argwhere(island[ : ,left_row] == 0).size > 0:
                        if k == 0:
                            probn = 0
                            empty_ind = np.argwhere(island[ : ,left_row] == 0)[0][0]
                        else:
                            # 当不是中间排时，计算当前岛当前纵排的所有空泊位中，阻挡概率最小的
                            probn_ls = [1 - (1/2 * (np.e**(- (event - isl_arrive[ : ,left_row][a][0]) / mu))) for a in np.argwhere(island[ : ,left_row] == 0)]
                            probn = min(probn_ls)
                            ind_ls = [a[0] for a in np.argwhere(island[ : ,left_row] == 0)]
                            empty_ind = ind_ls[np.argmin(probn)]
                        island[empty_ind][left_row] = veh
                        isl_arrive[empty_ind][left_row] = event
                        is_parked = 1
                        break
                    # 如果左侧没有空位，检测右侧是否有空位
                    elif np.argwhere(island[ : ,right_row] == 0).size > 0:
                        if k == 0:
                            empty_ind = np.argwhere(island[ : ,right_row] == 0)[0][0]
                        else:
                            probn_ls = [1 - (1/2 * (np.e**(- (event - isl_arrive[ : ,right_row][a][0]) / mu))) for a in np.argwhere(island[ : ,right_row] == 0)]
                            probn = min(probn_ls)
                            ind_ls = [a[0] for a in np.argwhere(island[ : ,right_row] == 0)]
                            empty_ind = ind_ls[np.argmin(probn)]
                        island[empty_ind][right_row] = veh
                        isl_arrive[empty_ind][right_row] = event
                        is_parked = 1
                        break
                    # 如果都没有空位，则检测下一个停车岛的该排
                    else:
                        continue
            return "accept"
        else:
            return "reject"

def depart(veh, facility, show = False):
    # veh: the target vehilce which wanna go out
    # facility: the parking faiclity (layout object)
    # First, locate the veh
    relocate = 0
    search_turn = 0
    block = 0
    # 首先判断该到达车辆是否是tpark中重定位的车辆
    for i in range(len(facility.tpark_info)):
        if np.argwhere(facility.tpark_info[i] == veh).size > 0:
            # 如果在tpark中找到了该车，由于车辆排列，不可能出现block，因此直接清空车位出场。
            veh_arg = np.argwhere(facility.tpark_info[i] == veh)[0]
            facility.tpark_info[i][veh_arg[0]][veh_arg[1]] = 0
            relocate_veh = np.array([])
            return relocate_veh, relocate, block
    for i in range(len(facility.isl_info)):
        island =  facility.isl_info[i]
        if np.argwhere(island == veh).size > 0:
            if show:
                locate = np.argwhere(island == veh)[0]
                stack_ind = locate[0]
                spot_ind = locate[1]
            locate = np.argwhere(island == veh)[0]
            stack_ind = locate[0]
            spot_ind = locate[1]
            stack_2k = len(island[stack_ind])
            stack_k = len(island[stack_ind]) / 2
            if spot_ind == 0 or spot_ind == stack_2k - 1:
                # 如果在最外侧，直接走
                island[stack_ind][spot_ind] = 0
                relocate_veh = np.array([])
                return relocate_veh, relocate, block
            elif spot_ind <= stack_k - 1:
                # 如果在左侧栈堆
                front = island[stack_ind][ : spot_ind]
                back = island[stack_ind][spot_ind + 1: ]
                front_copy = front.copy()
                back_copy = back.copy()
                if len(np.where(front != 0)[0]) <= len(np.where(back != 0)[0]):
                    # 如果前方车辆比后面少，从前面走
                    if len(np.where(front_copy != 0)[0]) != 0:
                        # 如果有车辆重定位，则先将重定位车辆安放到临时停车道
                        # 检索临时停车道是否有空闲位置 i
                        # 如果有足够的空闲位置，则重定位车辆在其中按顺序停放
                        if facility.tpark_info[i].shape[0] * facility.tpark_info[i].shape[1] >= len(np.where(front_copy != 0)[0]):
                            args = np.argwhere(front_copy != 0)
                            relocate_veh = front_copy[args]
                            island[stack_ind][ : spot_ind] = [0 for i in range(len(front_copy))]
                            row = facility.tpark_info[i].shape[1]
                            lane = 0
                            for j in range(0, relocate_veh.size, row):
                                count = 0
                                while count < row and j + count < relocate_veh.size:
                                    veh_arg = j + count
                                    this_veh = relocate_veh[veh_arg][0]
                                    facility.tpark_info[i][lane][count] = this_veh
                                    count += 1
                                lane += 1
                            relocate += len(relocate_veh)
                            island[stack_ind][spot_ind] = 0 # 目标车辆离场
                            return relocate_veh, relocate, block
                        # 如果没有足够的空闲位子，先移动可以移动的车辆，返回该部分移动过的车辆，并且没有移动过的车辆&需要离场的车辆
                        else:
                            tpark_size = facility.tpark_info[i].shape[0] * facility.tpark_info[i].shape[1]
                            args = np.argwhere(front_copy != 0)
                            relocate_veh = front_copy[args][: tpark_size]
                            for r in relocate_veh:
                                relocate_args = np.argwhere(island[stack_ind] == r)
                                island[stack_ind][relocate_args] = 0
                            row = facility.tpark_info[i].shape[1]
                            lane = 0
                            # 首先将可以移动的车辆移动
                            for j in range(0, relocate_veh.size, row):
                                count = 0
                                while count < row and j + count < relocate_veh.size:
                                    veh_arg = j + count
                                    this_veh = relocate_veh[veh_arg][0]
                                    facility.tpark_info[i][lane][count] = this_veh
                                    count += 1
                                lane += 1
                            relocate += len(relocate_veh)
                            # 然后返回block标识，如果有block标识，则event中首先加入relocate_veh的内容，然后relocate后，block_veh再离场
                            block = 1
                            return relocate_veh, relocate, block
                    else:
                        # 如果没有车辆被重定位
                        args = np.argwhere(front_copy != 0)
                        relocate_veh = front_copy[args] # 此时为空
                        island[stack_ind][spot_ind] = 0 # 目标车辆离场
                    return relocate_veh, relocate, block
                else:
                    # 如果后方车辆比前面少，从后面走，此时重定位移动到的是i+1 tpark
                    if len(np.where(back_copy != 0)[0]) != 0:
                        # 如果后方车辆存在，则需要进行重定位
                        # 检索临时停车道是否有空闲位置
                        # 如果有足够的空闲位置，则重定位车辆在其中按顺序停放
                        if facility.tpark_info[i + 1].shape[0] * facility.tpark_info[i + 1].shape[1] >= len(np.where(back_copy != 0)[0]):
                            args = np.argwhere(back_copy != 0)
                            relocate_veh = back_copy[args]
                            island[stack_ind][spot_ind + 1 : ] = [0 for i in range(len(back_copy))]
                            row = facility.tpark_info[i + 1].shape[1]
                            lane = 0
                            for j in range(0, relocate_veh.size, row):
                                count = 0
                                while count < row and j + count < relocate_veh.size:
                                    veh_arg = j + count
                                    this_veh = relocate_veh[veh_arg][0]
                                    facility.tpark_info[i + 1][lane][count] = this_veh
                                    count += 1
                                lane += 1
                            relocate += len(relocate_veh)
                            island[stack_ind][spot_ind] = 0 # 目标车辆离场
                            return relocate_veh, relocate, block
                        # 如果没有足够的空闲位子，先移动可以移动的车辆，返回该部分移动过的车辆，并且没有移动过的车辆&需要离场的车辆
                        else:
                            args = np.argwhere(back_copy != 0)
                            tpark_size = facility.tpark_info[i + 1].shape[0] * facility.tpark_info[i + 1].shape[1]
                            relocate_veh = back_copy[args][len(back_copy[args]) - 1 - tpark_size : ]
                            for r in relocate_veh:
                                relocate_args = np.argwhere(island[stack_ind] == r)
                                island[stack_ind][relocate_args] = 0
                            row = facility.tpark_info[i + 1].shape[1]
                            lane = 0
                            # 首先将可以移动的车辆移动
                            for j in range(0, relocate_veh.size, row):
                                count = 0
                                while count < row and j + count < relocate_veh.size:
                                    veh_arg = j + count
                                    this_veh = relocate_veh[veh_arg][0]
                                    facility.tpark_info[i + 1][lane][count] = this_veh
                                    count += 1
                                lane += 1
                            relocate += len(relocate_veh)
                            # 然后返回block标识，如果有block标识，则event中首先加入relocate_veh的内容，然后relocate后，block_veh再离场
                            block = 1
                            return relocate_veh, relocate, block
                    else:
                        # 如果没有车辆被重定位
                        args = np.argwhere(back_copy != 0)
                        relocate_veh = back_copy[args] # 此时为空
                        island[stack_ind][spot_ind] = 0 # 目标车辆离场
                        return relocate_veh, relocate, block
            elif spot_ind >= stack_k:
                # 如果在右侧栈堆
                back = island[stack_ind][ : spot_ind]
                front = island[stack_ind][spot_ind + 1: ]
                front_copy = front.copy()
                back_copy = back.copy()
                if len(np.where(front != 0)[0]) <= len(np.where(back != 0)[0]):
                    # 如果前方车辆比后面少，从前面走
                    if len(np.where(front_copy != 0)[0]) != 0:
                        # 如果有车辆重定位，则先将重定位车辆安放到临时停车道
                        # 检索临时停车道是否有空闲位置,此时是i + 1tpark
                        # 如果有足够的空闲位置，则重定位车辆在其中按顺序停放
                        args = np.argwhere(front_copy != 0)
                        if facility.tpark_info[i + 1].shape[0] * facility.tpark_info[i + 1].shape[1] >= len(np.where(front_copy != 0)[0]):
                            relocate_veh = front_copy[args]
                            island[stack_ind][spot_ind + 1 :] = [0 for i in range(len(front_copy))]
                            row = facility.tpark_info[i + 1].shape[1]
                            lane = 0
                            for j in range(0, relocate_veh.size, row):
                                count = 0
                                while count < row and j + count < relocate_veh.size:
                                    veh_arg = j + count
                                    this_veh = relocate_veh[veh_arg][0]
                                    facility.tpark_info[i + 1][lane][count] = this_veh
                                    count += 1
                                lane += 1
                            relocate += len(relocate_veh)
                            island[stack_ind][spot_ind] = 0 # 目标车辆离场
                            return relocate_veh, relocate, block
                        # 如果没有足够的空闲位子，先移动可以移动的车辆，返回该部分移动过的车辆，并且没有移动过的车辆&需要离场的车辆
                        else:
                            args = np.argwhere(front_copy != 0)
                            tpark_size = facility.tpark_info[i + 1].shape[0] * facility.tpark_info[i + 1].shape[1]
                            relocate_veh = front_copy[args][len(front_copy[args]) - tpark_size : ]
                            for r in relocate_veh:
                                relocate_args = np.argwhere(island[stack_ind] == r)
                                island[stack_ind][relocate_args] = 0
                            row = facility.tpark_info[i + 1].shape[1]
                            lane = 0
                            # 首先将可以移动的车辆移动
                            for j in range(0, relocate_veh.size, row):
                                count = 0
                                while count < row and j + count < relocate_veh.size:
                                    veh_arg = j + count
                                    this_veh = relocate_veh[veh_arg][0]
                                    facility.tpark_info[i + 1][lane][count] = this_veh
                                    count += 1
                                lane += 1
                            relocate += len(relocate_veh)
                            # 然后返回block标识，如果有block标识，则event中首先加入relocate_veh的内容，然后relocate后，block_veh再离场
                            block = 1
                            return relocate_veh, relocate, block
                    else:
                        # 如果没有车辆被重定位
                        args = np.argwhere(front_copy != 0)
                        relocate_veh = front_copy[args] # 此时为空
                        island[stack_ind][spot_ind] = 0 # 目标车辆离场
                        return relocate_veh, relocate, block
                else:
                    # 如果后方车辆比前面少，从后面走，此时重定位移动到的是i tpark
                    if len(np.where(back_copy != 0)[0]) != 0:
                        # 如果后方车辆存在，则需要进行重定位
                        # 检索临时停车道是否有空闲位置
                        # 如果有足够的空闲位置，则重定位车辆在其中按顺序停放
                        if facility.tpark_info[i].shape[0] * facility.tpark_info[i].shape[1] >= len(np.where(back_copy != 0)[0]):
                            args = np.argwhere(back_copy != 0)
                            relocate_veh = back_copy[args]
                            island[stack_ind][ : spot_ind] = [0 for i in range(len(back_copy))]
                            row = facility.tpark_info[i].shape[1]
                            lane = 0
                            for j in range(0, relocate_veh.size, row):
                                count = 0
                                while count < row and j + count < relocate_veh.size:
                                    veh_arg = j + count
                                    this_veh = relocate_veh[veh_arg][0]
                                    facility.tpark_info[i][lane][count] = this_veh
                                    count += 1
                                lane += 1
                            relocate += len(relocate_veh)
                            island[stack_ind][spot_ind] = 0 # 目标车辆离场
                            return relocate_veh, relocate, block
                        # 如果没有足够的空闲位子，先移动可以移动的车辆，返回该部分移动过的车辆，并且没有移动过的车辆&需要离场的车辆
                        else:
                            args = np.argwhere(back_copy != 0)
                            tpark_size = facility.tpark_info[i].shape[0] * facility.tpark_info[i].shape[1]
                            relocate_veh = back_copy[args][tpark_size : ]
                            for r in relocate_veh:
                                relocate_args = np.argwhere(island[stack_ind] == r)
                                island[stack_ind][relocate_args] = 0
                            row = facility.tpark_info[i].shape[1]
                            lane = 0
                            # 首先将可以移动的车辆移动
                            for j in range(0, relocate_veh.size, row):
                                count = 0
                                while count < row and j + count < relocate_veh.size:
                                    veh_arg = j + count
                                    this_veh = relocate_veh[veh_arg][0]
                                    facility.tpark_info[i][lane][count] = this_veh
                                    count += 1
                                lane += 1
                            relocate += len(relocate_veh)
                            # 然后返回block标识，如果有block标识，则event中首先加入relocate_veh的内容，然后relocate后，block_veh再离场
                            block = 1
                            return relocate_veh, relocate, block
                    else:
                        # 如果没有车子被重定位
                        args = np.argwhere(back_copy != 0)
                        relocate_veh = back_copy[args] # 此时为空
                        island[stack_ind][spot_ind] = 0 # 目标车辆离场
                        return relocate_veh, relocate, block
        # 如果车辆不在当前停车岛，继续寻找下一个停车岛
        search_turn += 1
    if search_turn >= len(facility.isl_info):
        # 如果找遍了所有停车岛都没有找到目标车辆
        # 先从tpark中找
        count = 0
        relocate = 0
        block = 0
        relocate_veh = np.array([])
        for i in range(len(facility.tpark_info)):
            if np.argwhere(facility.tpark_info[i] == veh).size > 0:
                # 如果在tpark中找到了该车，由于车辆排列，不可能出现block，因此直接清空车位离场。
                veh_arg = np.argwhere(facility.tpark_info[i] == veh)
                facility.tpark_info[i][veh_arg][0] = 0
                return relocate_veh, relocate, block
            count += 1
        if count >= len(facility.tpark_info):
            # print(veh, facility.tpark_info, facility.isl_info)
            raise ValueError("Cannot find this vehicle")



def simulation(facility, event, ind, dwell, demand, policy = "C2"):
    turn = 0
    relocate = 0
    reject = 0
    block = 0
    start = 0
    end = 0
    history = []
    start_time = time.time()
    initial_event = event.copy()
    while start == 0 or end == 0:
        # 第一步，判断该车辆是到达还是离去
        start = 1
        e = event[turn]
        v = ind[turn]
        if turn % 40 == 0:
            pass
            # print(facility.isl_info)
        if v <= demand:
            # 如果是到达，则调用到达函数，分配车辆
            d = dwell[v - 1]
            if time.time() - start_time > 2:
                show = True
            else:
                show = False
            flag = arrive(v, turn, facility, d, e, policy, show)
            if flag == "reject":
                # 如果reject，则reject+1
                # 并且，将其从离场中去除
                reject += 1
                depart_index = np.argwhere(ind == v + demand)[0][0]
                ind = np.delete(ind, depart_index)
                event = np.delete(event, depart_index)
        elif v > demand:
            # 如果是离去，则调用离去函数，车辆离场，并返回重定位的车辆id
            if time.time() - start_time > 2:
                show = True
            else:
                show = False
            relocate_veh, relocate_flag, block_flag = depart(v-demand, facility, show)
            relocate += relocate_flag
            if len(relocate_veh) > 0:
                # 如果存在重定位的车辆
                # 先让重定位的车辆重定位
                relocate_veh = relocate_veh.flatten()
                    # raise ValueError("None")
                relocate_event = [initial_event[int(v)] for v in relocate_veh]
                event = np.insert(event, turn + 1, relocate_event)
                ind = np.insert(ind, turn + 1, relocate_veh)
                history.extend(relocate_veh)
                # 然后判断是否存在block
                if block_flag == 1:
                    block += 1
                    # 如果存在block，需要在event中再插入当前veh
                    event = np.insert(event, turn + 1 + len(relocate_veh), e)
                    ind = np.insert(ind, turn + 1 + len(relocate_veh), v)
        # 判断是否已经结束（是否已经清空）
        spare_spot = sum([len(np.argwhere(facility.isl_info[i] == 0)) for i in range(len(facility.isl_info))])
        end_time = time.time()
        last_time = end_time - start_time
        # if last_time >= 5:
            # 超过5s还没有运行完毕，则说明遇到了不可行布局：
            # 在不可行布局下，临时停放车道容量太小，无法容纳车辆，造成阻断
            # 并且，在停车场满时，造成阻断的车辆分批重定位无法解决问题，其重定位只可以选择其出场的位置，造成永远的阻断。
            # return 0, 0, 0, 10**20
        if spare_spot == facility.get_capacity():
            end = 1
        turn += 1
    return relocate, reject, block, 0

# event, ind, demand, dwell_type = generate(1/120, 4, 400, 4)
# num_stack = int(np.floor((20 - 3) / 2.5))
# facility = Layout(16, [2,2,4,3,7,7,7,7,4,2,3,1,1,1,1,1], [1 for i in range(17)], 1, num_stack)
# relocate, reject, block, feasible = simulation(facility, event, ind, dwell_type, demand, policy = "C1")
# print(relocate, reject, block, feasible, facility.get_capacity(),facility.get_size())
