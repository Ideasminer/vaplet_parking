import numpy as np
import matplotlib.pyplot as plt
from simulation import *
import pandas as pd
import time

def variable2str(variable):
    key = ''
    for e in variable[0]:
        key += str(e)
    key += '_'
    for q in variable[1]:
        key += str(q)
    return key

def memory(variable, heap, value = None, mode = "save"):
    key = variable2str(variable)
    if mode == "save":
        dictionary = {key : value}
        if not heap.get(key) is None:
            if heap.get(key) != value:
                heap.update(**dictionary)
        else:
            heap.update(**dictionary)
        return heap
    if mode == "get":
        return heap.get(key)


class GA():
    def __init__(self, lb, ub, var_num, capacity, population, mu, pc, maxit, flow, size):
        # lb 和 ub是停车岛规模的上下限，下限一般为1，上限根据停车场尺寸决定
        # population 为种群大小
        # var_num 是停车场最多有几个停车岛
        # mu 是变异概率
        # pc 是交叉率
        # cost_func 是适应度函数
        # maxit 是最大迭代次数
        # pop 是种群
        # history用来记录最优个体和其适应度值
        self.lb = lb
        self.ub = ub
        self.var_num = var_num
        self.population = population
        self.mu = mu
        self.pc = pc
        self.maxit = maxit
        self.capacity = capacity
        self.pop = [[], []]
        self.bestsol = None
        self.bestres = np.inf
        self.history = [[], []]
        self.heap = {}
        self.size = size
        self.flow = flow


    def encode(self, var_num, ub):
        # 二进制编码，将var_num个停车岛看作一条染色体
        # 每个停车岛的信息是其中的一小段
        # 每段的长度根据ub确定，如20 = 10100，15 = 1111，只看最高位， 2^5 > 20 > 2^4, 15 < 2^4
        chromosome = []
        max_digit = 0
        while 2 ** (max_digit + 1) <= ub:
            max_digit += 1
        for i in range(var_num):
            chromosome.extend(np.random.randint(0, 2, max_digit + 1))
        return chromosome

    def decode(self, chromosome, var_num):
        # 二进制解码成为实数变量
        max_digit = len(chromosome) // var_num
        variable = []
        for i in range(var_num):
            start_digit = i * max_digit
            end_digit = (i + 1) * max_digit
            gene = chromosome[start_digit : end_digit]
            value = sum([gene[i] * (2 ** (max_digit - i - 1)) for i in range(len(gene))])
            variable.append(value)
        return variable

    def recode(self, ls, max_digit):
        # 根据显形返回基因型
        chromosome = []
        for i in range(len(ls)):
            gene = []
            val = ls[i]
            counter = 0
            while val // 2 != 0:
                counter += 1
                gene.append(val % 2)
                val = val // 2
            if counter == 0:
                # 如果没有进入上述循环，即value = 1或者0
                gene.append(val % 2)
            if len(gene) < max_digit:
                # 如果基因长度小于要求，则进行扩充
                diff = max_digit - len(gene)
                temp = [0 for j in range(diff)]
                temp.extend(gene)
                gene = temp
            chromosome.extend(gene)
        return chromosome

    def initialize(self):
        # 初始化，生成两类变量
        # 第一类变量为停车岛的布局（大小），其直接encode即可
        # 第二类变量为临时停车道的布局（大小），其数量总为var_num + 1
        # 将两类变量分别存储在pop的两个数组中
        population = self.population
        var_num = self.var_num
        y_var_num = var_num + 1
        # 第二类变量的上限最大为(ub - 1) / capacity
        y_ub = np.ceil((self.ub - 1) // self.capacity)
        for i in range(population):
            x_chromosome = self.encode(var_num, self.ub)
            y_chromosome = self.encode(y_var_num, y_ub)
            self.pop[0].append(x_chromosome)
            self.pop[1].append(y_chromosome)
        return self.pop

    def crossover(self, p1, p2):
        p1x = self.pop[0][p1]
        p1y = self.pop[1][p1]
        p2x = self.pop[0][p2]
        p2y = self.pop[1][p2]
        x_cross_point = len(p1x) // 2
        c1x = p1x[:x_cross_point].copy()
        c1x.extend(p2x[x_cross_point:])
        c2x = p2x[:x_cross_point].copy()
        c2x.extend(p1x[x_cross_point:])
        y_cross_point = len(p1y) // 2
        c1y = p1y[:y_cross_point].copy()
        c1y.extend(p2y[y_cross_point:])
        c2y = p2y[:y_cross_point].copy()
        c2y.extend(p1y[y_cross_point:])
        c1 = [c1x, c1y]
        c2 = [c2x, c2y]
        return c1, c2

    def mutate(self, chromosome):
        mu = self.mu
        chrome_length = len(chromosome)
        alter_args = np.argwhere(np.random.uniform(0, 1, chrome_length) <= mu)
        for i in alter_args:
            digit = chromosome[i[0]]
            if digit == 1:
                chromosome[i[0]] = 0
            elif digit == 0:
                chromosome[i[0]] = 1
        return chromosome

    def repair(self):
        # 对不符合约束的染色体进行修复
        # 约束为：
            # 如果两端的停车道不存在（解码值为0），则第一个/最后一个临时停放车道不存在
            # 对于非两端的临时停放车道，如果其前面一个停车岛不存在（且前面一个停车岛不是边缘停车岛），则其不存在
            # 如果临时停放车道存在，则判断是否其满足上下限
            # 如果超过了上限，则进行修复，从最低位修复为0，重新判断，直到进入约束范围内
            # 由于下限一般为0（不存在），且已经对是否存在进行过修复，因此一般不会低于下限值
        for i in range(self.population):
            x_chromosome = self.pop[0][i]
            y_chromosome = self.pop[1][i]
            x_max_digit = len(x_chromosome) // self.var_num
            y_max_digit = len(y_chromosome) // (self.var_num + 1)
            x_variable = self.decode(x_chromosome, self.var_num)
            y_variable = self.decode(y_chromosome, self.var_num + 1)
            # 第一步，要保证至少有一个停车岛存在，且不超过规模上限
            if sum(x_variable) == 0:
                # 如果一个停车岛都没有，则要生成停车岛，这里修复的方法为，让其中一个停车岛的规模变为1，这样既保证了存在停车岛，同时保证了不超过规模上限
                temp = self.pop[0][i][ : x_max_digit]
                temp[-1] = 1
                self.pop[0][i][ : x_max_digit] = temp
                x_chromosome = self.pop[0][i]
                x_variable = self.decode(x_chromosome, self.var_num)
            # 第二步，修复第一类变量（停车岛规模），判断其是否超过上限
            for n in range(len(x_variable)):
                x_position = x_max_digit - 1
                while x_variable[n] > self.ub and x_position >= 0:
                    self.pop[0][i][n * x_max_digit : (n + 1) * x_max_digit][x_position] = 0
                    x_chromosome = self.pop[0][i]
                    x_variable = self.decode(x_chromosome, self.var_num)
                    x_position -= 1
            # 第三步，判断临时停放车道是否存在
            for j in range(len(x_variable)):
                if j == 0:
                    # 如果是第一个停车岛
                    if x_variable[j] == 0:
                        # 如果第一个停车岛不存在，则第一个临时停放车道也不存在
                        self.pop[1][i][ : y_max_digit] = [0 for i in range(y_max_digit)]
                        y_variable = self.decode(self.pop[1][i], self.var_num + 1)
                    else:
                        # 如果第一个停车岛存在，判断是否第一个临时停放车道存在
                        while y_variable[j] == 0:
                            # 如果第一个临时停放车道不存在，则对该临时停放车道重新初始化，直到其存在为止。
                             self.pop[1][i][ : y_max_digit] = np.random.randint(0, 2, y_max_digit)
                             y_variable = self.decode(self.pop[1][i], self.var_num + 1)
                if j == len(x_variable) - 1:
                    # 如果是最后一个停车岛
                    if x_variable[j] == 0:
                        # 如果最后一个停车岛不存在，则最后一个临时停车道也不存在
                        self.pop[1][i][len(self.pop[1][i]) - y_max_digit : ] = [0 for i in range(y_max_digit)]
                        y_variable = self.decode(self.pop[1][i], self.var_num + 1)
                    else:
                        while y_variable[-1] == 0:
                            # 如果最后一个临时停放车道不存在，则对该临时停放车道重新初始化，直到其存在为止。
                             self.pop[1][i][len(self.pop[1][i]) - y_max_digit : ] = np.random.randint(0, 2, y_max_digit)
                             y_variable = self.decode(self.pop[1][i], self.var_num + 1)
                else:
                    # 如果不是末端的临时停放车道，则看其下一个停车岛是否存在
                    # 如果下一个停车岛不存在，且不为最后一个停车岛，则当前临时停放车道不存在
                    if x_variable[j + 1] == 0 and j + 1 != len(x_variable) - 1:
                        self.pop[1][i][(j + 1) * y_max_digit : (j + 2) * y_max_digit] = [0 for i in range(y_max_digit)]
                        y_variable = self.decode(self.pop[1][i], self.var_num + 1)
                    # 如果下一个停车岛存在，则当前临时停放车道存在
                    if x_variable[j + 1] != 0:
                        while y_variable[j + 1] == 0:
                            self.pop[1][i][(j + 1) * y_max_digit : (j + 2) * y_max_digit] = np.random.randint(0, 2, y_max_digit)
                            y_variable = self.decode(self.pop[1][i], self.var_num + 1)
                    # 如果下一个停车岛是最后一个停车岛，无论其前一个岛存不存在，下一个岛存不存在，其必然存在，则该临时停放车道存在
                    if j + 1 == len(x_variable) - 1:
                        while y_variable[j + 1] == 0:
                            self.pop[1][i][(j + 1) * y_max_digit : (j + 2) * y_max_digit] = np.random.randint(0, 2, y_max_digit)
                            y_variable = self.decode(self.pop[1][i], self.var_num + 1)
            # 第四步，判断每一个临时停放车道的规模是否超过上界
            y_ub = []
            y_ub.append(np.ceil((x_variable[0] - 1) / self.capacity))
            for m in range(1, len(x_variable)):
                y_ub.append(np.max([np.ceil((x_variable[m - 1] -1) / self.capacity),np.ceil((x_variable[m]  - 1) / self.capacity)]))
            y_ub.append(np.ceil((x_variable[-1] - 1) / self.capacity))
            for k in range(len(y_variable)):
                y_position = y_max_digit - 1
                while y_variable[k] > y_ub[k] and y_position >= 0:
                    # 如果超过了上界，则y染色体从最后一位开始变为0，直到到达约束范围内
                    # 此方法必然可以修复到约束范围内，极端情况为全部修复到0
                    self.pop[1][i][k * y_max_digit : (k + 1) * y_max_digit][y_position] = 0
                    y_variable = self.decode(self.pop[1][i], self.var_num + 1)
                    y_position -= 1
        return self.pop

    def local(self):
        # 进行局域搜索
        # 搜索的逻辑为：
            # 保持停车岛总的规模不变，调整其组合方式，如[1, 2, 5] => [4, 1, 3]
            # 要保持总的规模不变，即保持所有位加起来的总和不变，移动则是相同位的加和移动，如果加和之后到达2，则进位。
            # 总是只移动最小位，以防止超过界限。（即变动大小为1）
            # 若有var_num个变量，则首先检测最小位是否有1，然后检测是否最低为有0，将所有1 -> 0, 0 -> 1
        bestsol = self.bestsol.copy()
        x_var_num = self.var_num
        chromosome = self.recode(bestsol[0], len(self.pop[0][0]) // x_var_num)
        chromosome_y = self.recode(bestsol[1], len(self.pop[1][0]) // (x_var_num + 1))
        x_max_digit = len(chromosome) // x_var_num
        wait_ls_0 = []
        wait_ls_1 = []
        for i in range(x_var_num):
            current_chromosome = chromosome[i * x_max_digit : (i + 1) * x_max_digit]
            if current_chromosome[-1] == 0:
                wait_ls_0.append(i)
            if current_chromosome[-1] == 1:
                wait_ls_1.append(i)
            if len(wait_ls_0) == 1 and len(wait_ls_1) == 1:
                break
        if len(wait_ls_0) == 1 and len(wait_ls_1) == 1:
            zero_ind = wait_ls_0[0]
            one_ind = wait_ls_1[0]
            chromosome[zero_ind * x_max_digit : (zero_ind + 1) * x_max_digit][-1] = 1
            chromosome[one_ind * x_max_digit : (one_ind + 1) * x_max_digit][-1] = 0
            self.pop[0].append(chromosome)
            self.pop[1].append(chromosome_y)
        self.repair()

    def select(self):
        # 选择较好的个体
        pop_val = [[self.decode(self.pop[0][i], self.var_num),
            self.decode(self.pop[1][i], self.var_num + 1)] for i in range(self.population)]
        val = [self.cost_func(x) for x in pop_val]
        # 进行轮盘赌算法的选择，同时需要保证至少有2个被选到，否则将难以进行crossover
        selected = []
        # 如果被选中的个数小于2个个体，则继续循环，直到选择大于等于2个个体
        while len(selected) < 2:
            prob = [x / sum(val) for x in val]
            prob = np.cumsum(prob)
            rand_array = np.random.rand(len(val))
            rand_array = sorted(rand_array)
            for i in range(len(rand_array)):
                for j in range(len(prob)):
                    if rand_array[i] <= prob[j]:
                        # 如果当前的累计概率大于等于随机概率，则当前个体被选中
                        # 如果当前个体没有被选中，则检索下一个个体
                        # 如果有被选中的，则跳出内部循环，外部循环i更新随机数，用新的随机数进行检索
                        selected.append(pop_val[j])
                        break
        temp_pop = [[], []]
        for i in range(len(selected)):
            temp_pop[0].append(selected[i][0])
            temp_pop[1].append(selected[i][1])
        self.pop = temp_pop

    def cost_func(self, variable):
        heap = self.heap
        size = self.size
        event, ind, demand, dwell_type = self.flow
        value = self.heap.get(variable2str(variable))
        if value:
            cost = value[0]
            return cost
        k = variable[0]
        num_tpark = variable[1]
        copy = [k, num_tpark]
        k = [i for i in k if i!=0]
        num_tpark = [i for i in num_tpark if i != 0]
        num_stack = int(np.floor((size[1] - 3) / 2.5))
        num_isl = len(k)
        try:
            facility = Layout(num_isl, k, num_tpark, 1, num_stack)
        except:
            raise ValueError(copy)
        try:
            width = facility.get_size()[0]
            height = facility.get_size()[1]
        except:
            raise ValueError(copy)
        area_diff = np.abs((width * height) - (size[0] * size[1]))
        relocate, reject, block, feasible = simulation(facility, event, ind, dwell_type, demand)
        facility_capacity = facility.get_capacity()
        cost = ((area_diff)** 2 + 1) * (relocate + 10 * reject + 5 * block + 1) + feasible
        cost = (10 ** 10) / cost
        memory(variable, self.heap, [cost, [area_diff, relocate, reject, block, facility_capacity]], mode = "save")
        return cost

    def run(self):
        # 运行整个流程
        # 第一步，进行初始化，并进行修复，保证在约束范围内
        self.initialize()
        self.repair()
        # 第二步，计算当前最好的个体
        pop_val = [[self.decode(self.pop[0][i], self.var_num),
            self.decode(self.pop[1][i], self.var_num + 1)] for i in range(len(self.pop[1]))]
        self.bestsol = sorted(pop_val, key = lambda x: self.cost_func(x))[0]
        self.bestres = self.cost_func(self.bestsol)
        self.history[0].append(variable2str(self.bestsol))
        self.history[1].append(self.bestres)
        best_ind = 0
        for val in range(len(pop_val)):
            if val == self.bestsol:
                break
            else:
                best_ind += 1
        for n in range(self.maxit):
            # 第三步，如果当前不是第一次进入该循环，进行局域搜索
            if len(self.history[1]) > 1:
                if self.history[1][-1] < self.history[1][-2]:
                    # 当之前的优化有效果时，在最优个体附近所有
                    self.local()
            # 第四步，进行交叉
            nc = int(np.round(self.pc * self.population / 2) * 2)
            for i in range(nc // 2):
                parent_array = np.random.permutation(self.population)
                # 最好的不交叉
                while best_ind in parent_array[:2]:
                    parent_array = np.random.permutation(len(self.pop))
                p1, p2 = parent_array[0], parent_array[1]
                c1, c2 = self.crossover(p1, p2)
                self.pop[0].append(c1[0])
                self.pop[0].append(c2[0])
                self.pop[1].append(c1[1])
                self.pop[1].append(c2[1])
            # 第五步，进行变异，最好的不变异
            [self.mutate(self.pop[0][c]) for c in range(len(self.pop[0])) if c != best_ind]
            [self.mutate(self.pop[1][c]) for c in range(len(self.pop[1])) if c != best_ind]
            # 第六步，进行修复
            self.repair()
            # 第七步，进行选择、淘汰
            self.select()
            # 第八步，重新计算最好的个体
            pop_val = [[self.decode(self.pop[0][i], self.var_num),
                self.decode(self.pop[1][i], self.var_num + 1)] for i in range(len(self.pop[1]))]
            self.bestsol = sorted(pop_val, key = lambda x: self.cost_func(x))[0]
            self.bestres = self.cost_func(self.bestsol)
            self.history[0].append(variable2str(self.bestsol))
            self.history[1].append(self.bestres)
        X = np.array([i + 1 for i in range(len(self.history[1]))])
        Y =  np.array(self.history[1])
        data = pd.DataFrame(self.history)
        data = data.T
        data.columns = ['sol', 'val']
        data.sort_values(by=['val'], inplace = True)
        bestsol, bestres = data.values[-1]
        area_diff = self.heap.get(bestsol)[1][0]
        relocate = self.heap.get(bestsol)[1][1]
        reject = self.heap.get(bestsol)[1][2]
        block = self.heap.get(bestsol)[1][3]
        facility_capacity = self.heap.get(bestsol)[1][4]
        return bestsol, bestres, area_diff, relocate, reject, block, facility_capacity


def pipeline(lb, ub, var_num, capacity, population, mu, pc, maxit, arrive, dwell, demand, size, times):
    history = [[],[], [], [], [], [], []]
    flow = generate(arrive, dwell, demand)
    for t in range(times):
        problem = GA(lb = lb, ub = ub, var_num = var_num,
        capacity = capacity, population = population, mu = mu,
        pc = pc, maxit = maxit, flow = flow, size = size)
        # print(problem)
        sol, res, area_diff, relocate, reject, block, facility_capacity = problem.run()
        history[0].append(sol)
        history[1].append(res)
        history[2].append(area_diff)
        history[3].append(relocate)
        history[4].append(reject)
        history[5].append(block)
        history[6].append(facility_capacity)
    data = pd.DataFrame(history)
    data = data.T
    data.columns = ['sol', 'res', 'area_diff', 'relocate', 'reject', 'block', 'facility_capacity']
    group = data.groupby(['sol'])
    out = group.count()
    out = pd.DataFrame(out)
    out.reset_index(inplace=True)
    out.sort_values(by = 'res', inplace = True)
    bestsol, bestres, best_area_diff, best_relocate, best_reject, best_block, best_capacity = out.values[-1]
    if bestres == 1:
        maxout = group.max()
        maxout = pd.DataFrame(maxout)
        maxout.reset_index(inplace=True)
        maxout.sort_values(by = 'res', inplace = True)
        bestsol, bestres, best_area_diff, best_relocate, best_reject, best_block, best_capacity = maxout.values[-1]
    return bestsol, bestres, best_area_diff, best_relocate, best_reject, best_block,best_capacity, data


# # # event, ind, demand, dwell_type = generate( 1 / 120, 4, 400)
# size = (40, 120)
# # heap = {}
# var_num = (size[0] - 5) // (10 + 5)
# capacity = int(np.floor((size[1] - 3) / 5))

# pipeline(lb = 0, ub = 7, var_num = var_num, capacity = capacity,
#                 population = 50, mu = 0.05, pc = 0.4,
#                 maxit = 30, arrive = 1 / 120, dwell = 4, demand = 400, size = (40, 120), times = 20)

# print(bestsol, bestres, best_area_diff, best_relocate, best_reject, best_block, data)
