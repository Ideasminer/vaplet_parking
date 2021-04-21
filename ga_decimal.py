import numpy as np
import matplotlib.pyplot as plt


class GA:
    def __init__(self, population, num_xvars, x_ub, x_lb, y_capacity, mu, cost_func, pc, maxit):
        self.population = population
        self.num_xvars = num_xvars
        self.x_ub = x_ub
        self.x_lb = x_lb
        self.y_capacity = y_capacity
        self.mu = mu
        self.cost_func = cost_func
        self.pc = pc
        self.maxit = maxit
        self.pop = []
        self.bestsol = None
        self.bestres = np.inf


    def initial_pop(self):
        # 产生初始种群
        for p in range(self.population):
            # 对于X类别的变量，其有固定的上下限，ub, lb
            # randint前闭后开
            flag = 0
            xvars = np.array([])
            while flag == 0 or np.sum(xvars) == 0:
                # xvars不可以全部为0
                xvars = np.random.randint(self.x_lb, self.x_ub + 1,self.num_xvars)
                flag = 1
            yvars = np.array([])
            # 对于Y类别变量，其上下限取决于对应位置的xi和xi+1的取值
            for i in range(0, len(xvars)):
                if i == 0:
                    # 如果是第一个
                    if xvars[i] == 0:
                        # 如果X类别变量第一个是0，则Y的第一个必然是0
                        yvars =  np.append(yvars, 0)
                    else:
                        # 如果X类别第一变量不是0，则Y可以取值
                        y_lb = 1
                        y_ub = np.ceil(np.max([xvars[i] - 1, 0]) / self.y_capacity)
                        if y_ub <= y_lb:
                            # 如果上界小于下界（即实际需要的最多的tpark小于1），则取下界，因为至少有1个
                            yvars = np.append(yvars, y_lb)
                        else:
                            # 如果上界大于下界，在两者之间取值
                            yvars = np.append(yvars, np.random.randint(y_lb, y_ub + 1))
                if i + 1 < len(xvars):
                    # 如果还没有到边界，遍历i-1次，产生i-1个
                    if xvars[i + 1] == 0 and i + 1 != len(xvars) - 1:
                        # 如果xi和xi+1都是0，则y为0
                        yvars =  np.append(yvars, 0)
                    else:
                        # 如果至少有一个不为0，则y有值
                        # 如果y有值，则y的下限是1， 上限是(xi * 2 - 1 + xi+1 * 2 - 1) / y_capacity
                        # 即认为是两侧栈堆同时有车出来
                        # 这里的y_capacity是确定值，指的是tpark一条车道的容量
                        y_lb = 1
                        y_ub = np.ceil(np.max([np.max([(xvars[i] - 1), 0]), 
                            np.max([(xvars[i + 1] - 1), 0])]) / self.y_capacity)
                        if y_ub <= y_lb:
                            yvars = np.append(yvars, y_lb)
                        else:
                            yvars = np.append(yvars, np.random.randint(y_lb, y_ub + 1))
                else:
                    # 如果是最后一个，则y的取值取决于xi有没有，如果xi不是0，则y的lb是1，ub取决于xi的大小
                    if xvars[i] == 0:
                        yvars =  np.append(yvars, 0)
                    else:
                        y_lb = 1
                        y_ub = np.ceil(np.max([xvars[i] - 1, 0]) / self.y_capacity)
                        if y_ub <= y_lb:
                            yvars = np.append(yvars, y_lb)
                        else:
                            yvars = np.append(yvars, np.random.randint(y_lb, y_ub + 1))
            self.pop.append(np.array([xvars, yvars]))
        


    def crossover(self, p1, p2):
        # p1 p2是两个父代
        c1 = p1.copy()
        c2 = p2.copy()
        x_alpha = np.random.uniform(0, 1, size=len(c1[0]))
        y_alpha = np.random.uniform(0, 1, size=len(c1[1]))
        c1[0] = np.round(p1[0] * x_alpha + p2[0] * (1 - x_alpha))
        c1[1] = np.round(p1[1] * y_alpha + p2[1] * (1 - y_alpha))
        c2[0] = np.round(p2[0] * x_alpha + p1[0] * (1 - x_alpha))
        c2[1] = np.round(p2[1] * y_alpha + p1[1] * (1 - y_alpha))
        c1[0] = np.array([int(x) for x in c1[0]])
        c1[1] = np.array([int(x) for x in c1[1]])
        c2[0] = np.array([int(x) for x in c2[0]])
        c2[1] = np.array([int(x) for x in c2[1]])
        return c1, c2

    
    def mutate(self, variable):
        variable_copy = variable.copy()
        x = variable_copy[0]
        y = variable_copy[1]
        flag_x = np.random.uniform(0, 1, size = len(x)) <= self.mu
        flag_y = np.random.uniform(0, 1, size = len(y)) <= self.mu
        x_arg = np.argwhere(flag_x)
        y_arg = np.argwhere(flag_y)
        x[x_arg] += np.random.randint(-self.x_ub, self.x_ub, size=x[x_arg].shape)
        y[y_arg] += np.random.randint(np.round(-(self.x_ub - 1) / self.y_capacity), np.round((self.x_ub - 1) / self.y_capacity), size = y[y_arg].shape)
        return variable_copy


    def constrain(self, variable):
        variable_copy = variable.copy()
        xvars = variable_copy[0]
        yvars = variable_copy[1]
        for j in range(len(xvars)):
            ub_val = np.min([xvars[j], self.x_ub])
            xvars[j] = ub_val
            lb_val = np.max([xvars[j], self.x_lb])
            xvars[j] = lb_val
        while np.sum(xvars) == 0:
            # 如果全部为0，不可以接受，则重新生成一个个体
            xvars = np.random.randint(self.x_lb, self.x_ub + 1,self.num_xvars)
        for i in range(0, len(xvars)):
            if i == 0:
                # 如果是第一个
                if xvars[i] == 0:
                    yvars[i] = 0
                else:
                    y_lb = np.max([yvars[i], 1])
                    yvars[i] = y_lb
                    y_ub = np.min([yvars[i], np.ceil(np.max([xvars[i] - 1, 0]) / self.y_capacity)])
                    yvars[i] = y_ub
                    if yvars[i] == 0:
                        yvars[i] = 1
            if i + 1 < len(xvars):
                # 如果还没有到边界，遍历i-1次，产生i-1个
                if xvars[i + 1] == 0 and i + 1 != len(xvars) - 1:
                    # 如果xi和xi+1都是0，则y为0
                    yvars[i + 1] = 0
                else:
                    # 如果至少有一个不为0，则y有值
                    # 如果y有值，则y的下限是1， 上限是(xi * 2 - 1 + xi+1 * 2 - 1) / y_capacity
                    # 即认为是两侧栈堆同时有车出来
                    # 这里的y_capacity是确定值，指的是tpark一条车道的容量
                    y_lb = np.max([yvars[i + 1], 1])
                    yvars[i + 1] = y_lb
                    y_ub = np.min([yvars[i + 1], np.ceil(np.max([np.max([(xvars[i] - 1), 0]), 
                        np.max([(xvars[i + 1] - 1), 0])]) / self.y_capacity)])
                    yvars[i + 1] = y_ub
                    if yvars[i + 1]  == 0:
                        yvars[i + 1] = 1
            else:
                # 如果是最后一个，则y的取值取决于xi有没有，如果xi不是0，则y的lb是1，ub取决于xi的大小
                if xvars[i] == 0:
                    yvars[i + 1] = 0
                else:
                    y_lb = np.max([yvars[i + 1], 1])
                    yvars[i + 1] = y_lb
                    y_ub = np.min([yvars[i + 1], np.ceil(np.max([xvars[i] - 1, 0]) / self.y_capacity)])
                    yvars[i + 1] = y_ub
                    if yvars[i + 1] == 0:
                        yvars[i + 1] = 1
        return variable_copy
    
    
    def select(self):
        self.pop = sorted(self.pop, key=lambda x: self.cost_func(x))
        # print("cost_func", [self.cost_func(x) for x in self.pop])
        self.pop = self.pop[:self.population]
        return self.pop


    def run(self):
        history = []
        # 初始化种群
        self.initial_pop()
        # 计算最优个体
        self.bestsol = sorted(self.pop, key=lambda x: self.cost_func(x))[0]
        self.bestres = self.cost_func(self.bestsol)
        for t in range(self.maxit):
            # 进行交叉
            nc = int(np.round(self.pc * self.population / 2) * 2)
            for i in range(nc // 2):
                parent_array = np.random.permutation(self.pop)
                p1, p2 = parent_array[0], parent_array[1]
                c1, c2 = self.crossover(p1, p2)
                self.pop.append(c1)
                self.pop.append(c2)
            # 进行变异
            self.pop = [self.mutate(x) for x in self.pop]
            # 约束
            self.pop = [self.constrain(x) for x in self.pop]
            # 选择
            self.select()
            # 输出
            self.bestsol = sorted(self.pop, key=lambda x: self.cost_func(x))[0]
            self.bestres = self.cost_func(self.bestsol)
            print(t, self.bestsol, self.bestres)
            history.append(self.bestres)
        X = np.array([i + 1 for i in range(len(history))])
        Y =  np.array(history)
        plt.plot(X, Y)
        plt.show()


from simulation import *
event, ind, demand, dwell_type = generate( 1 / 120, 4, 320)
size = (80, 60)
def cost_func(variable, size = size, event = event, ind = ind, demand = demand, dwell_type = dwell_type):
    k = variable[0].tolist()
    num_tpark = variable[1].tolist()
    k = [i for i in k if i!=0]
    num_tpark = [i for i in num_tpark if i != 0]
    num_stack = int(np.floor((size[1] - 3) / 2.5))
    num_isl = len(k)
    facility = Layout(num_isl, k, num_tpark, 1, num_stack)
    width = facility.get_size()[0]
    height = facility.get_size()[1]
    area_diff = np.abs(width * height - size[0] * size[1])
    relocate, reject, block = simulation(facility, event, ind, dwell_type, demand)
    cost = (area_diff) + (area_diff) * (relocate + 5 * reject + 3 * block)
    return cost


num_xvars = (size[0] - 5) // (10 + 5)
x_ub = (size[0] - 10) // 10
x_lb = 0
y_capacity = np.floor(size[1] - 3) // 5
problem = GA(30, num_xvars, x_ub, x_lb, y_capacity, 0.2, cost_func, 0.4, 100)
problem.run()

# k = [3, 4, 1, 1]
# num_tpark = [1, 1, 1, 1, 1]
# num_stack = int(np.floor((size[1] - 3) / 2.5))
# print(num_stack)
# facility = Layout(4, k, num_tpark, 1, num_stack)
# print(facility.get_size())