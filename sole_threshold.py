# 计算单一停车岛的规模阈值并出图

from simulation import Layout, simulation, generate
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from  sklearn.linear_model import LinearRegression
plt.rcParams['font.sans-serif'] = ['FangSong']
plt.rcParams['axes.unicode_minus'] = False
max_k = 20
num_stack = 10
lam = 1 / 120
mu = 4
threshold = 4

def generate_mod(lam, mu, start_demand, max_demand, threshold, capacity, step = 20, rate=0.9):
    for i in range(start_demand, max_demand + 1):
        event, ind, demand, dwell_type = generate(lam, mu, i, threshold)
        in_veh = 0
        out_veh = 0
        occupy = []
        for i in range(len(ind)):
            if ind[i] <= demand:
                # 当前车辆为到达车辆
                in_veh += 1
            else:
                out_veh += 1
            occupy.append(in_veh - out_veh)
        peak_occupy_rate = max(occupy)
        if peak_occupy_rate / capacity >= rate:
            # print(peak_occupy_rate / capacity)
            return event, ind, demand, dwell_type
    raise ValueError("Not Found!", peak_occupy_rate/capacity)

# Mian Loop
def get_res(max_k, num_stack, lam, mu, threshold):
    relocate_rate = []
    reject_rate = []
    block_rate = []
    for i in range(1, max_k + 1):
        # generate Layout
        num_isl = 1
        k = [i]
        sole_tpark = max(np.ceil((i - 1) * 5 / (num_stack * 2.5)), 1)
        tpark = [sole_tpark, sole_tpark]
        facility = Layout(num_isl, k, tpark, 1, num_stack)
        capacity = facility.get_capacity()
        # set peak demand = 80% capacity
        max_demand = int(2 * capacity)
        start_demand = int(0.5 * capacity)
        # generate vehilce
        event, ind, demand, dwell_type = generate_mod(lam, mu, start_demand, max_demand, threshold, capacity)
        in_veh = 0
        out_veh = 0
        occupy = []
        for i in range(len(ind)):
            if ind[i] <= demand:
                # 当前车辆为到达车辆
                in_veh += 1
            else:
                out_veh += 1
            occupy.append(in_veh - out_veh)
        peak_occupy = max(occupy)
        # print(peak_occupy / demand)
        relocate, reject, block, feasible = simulation(facility, event, ind, dwell_type, demand, policy = "C4")
        time_range = event[-1] - event[0]
        relocate_rate.append(relocate / demand)
        reject_rate.append(reject / demand)
        block_rate.append(block / demand)
    return relocate_rate, reject_rate, block_rate

def res_plot(max_k, res, name, folder = '.'):
    x=[i for i in range(1, 1 + max_k)]
    y=res
    plt.scatter(x, y, c='blue', marker = 'x')
    plt.title('C2策略下重定位率随停车岛规模的变化趋势')
    plt.xlabel('仿真次数')
    plt.ylabel('重定位率')

relocate_rate, reject_rate, block_rate = get_res(max_k, num_stack, lam, mu, threshold)
res_plot(max_k, relocate_rate, 'relocate_rate')
# for i, j in zip([relocate_rate, reject_rate, block_rate],['relocate_rate', 'reject_rate', 'block_rate']):
#     res_plot(max_k, i, j)

model = LinearRegression()
model.fit(np.array([i for i in range(max_k)]).reshape(-1, 1), np.array(relocate_rate).reshape(-1, 1))
x = [i for i in np.arange(1, 20.1, 0.01)]
y = model.predict(np.array(x).reshape(-1,1)).tolist()
plt.plot(x, y, c='red')
plt.legend(['趋势线'],loc='best')

plt.savefig('.' + '/' + 'single_{}.png'.format('relocate_rate'), dpi = 1000)
plt.close()
print(model.intercept_, model.coef_)
