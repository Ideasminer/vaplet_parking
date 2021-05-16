import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from simulation import *
import pandas as pd
from matplotlib import cm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
plt.rcParams['font.sans-serif'] = ['FangSong']
plt.rcParams['axes.unicode_minus'] = False

def get_res_ls(facility, lam, mu, threshold, num, maxit):
    policy = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2', 'C3', 'C4']
    relocate_data, reject_data, block_data = [], [], []
    for i in range(maxit):
        event, ind, demand, dwell_type = generate(lam, mu, num, threshold)
        relocate_ls, reject_ls, block_ls = [], [] ,[]
        for p in policy:
            relocate, reject, block, feasible = simulation(facility, event, ind, dwell_type, demand, policy = p)
            relocate_ls.append(relocate)
            reject_ls.append(reject)
            block_ls.append(block)
        relocate_data.append(relocate_ls)
        reject_data.append(reject_ls)
        block_data.append(block_ls)
    return relocate_data, reject_data, block_data

def plot_policy(ls, title=None):
    fig = plt.figure()
    ax = fig.gca()
    fig.subplots_adjust(left=0.13, right=0.87, top = 0.87, bottom=0.13)
    policy = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2', 'C3', 'C4']
    ax.bar(np.array(policy),np.array(ls),color=['red','blue','green','orange','black', 'violet','brown', 'yellow'])
    ax.set_title('各策略下平均'+title)
    ax.set_xlabel('策略')
    ax.set_ylabel(title)
    val = np.array(ls)
    if title == "重定位次数":
        ax.set_ylim(5 * (np.min(val) // 5 - 1), 5 * (np.max(val) // 5 + 1))
    plt.savefig('./{}.png'.format(title), dpi = 1000)
    plt.close()

def regression(x, y):
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    poly = PolynomialFeatures(degree=6)
    x = poly.fit_transform(x)
    model = LinearRegression()
    model.fit(x, y)
    return model

def regression2mean(ls, title, policy):
    ind = np.array(list(range(len(ls))))
    val = np.array(ls)
    fig = plt.figure()
    ax = fig.gca()
    fig.subplots_adjust(left=0.18, right=0.87, top = 0.87, bottom=0.13)
    scatter = ax.scatter(ind, val, color = 'blue', marker='x')
    ax.set_xlabel('仿真序号（次）', labelpad = 10)
    ax.set_ylabel(title, labelpad = 10)
    model = regression(ind, val)
    expand_x = np.linspace(np.min(ind), np.max(ind), 100).reshape(-1, 1)
    expand_x_copy = expand_x.copy()
    poly = PolynomialFeatures(degree=6)
    expand_x = poly.fit_transform(expand_x)
    expand_y = model.predict(expand_x).reshape(-1, 1)
    line = ax.plot(expand_x_copy, expand_y, c='red')
    ax.legend(line, ['趋势线'])
    plt.axhline(np.array(ls).mean(), label='mean',linestyle='-.', color='r')
    ax.set_title('{}策略-{}波动图'.format(policy,title))
    if title == "重定位次数":
        plt.ylim(5 * (np.min(val) // 5 - 1), 5 * (np.max(val) // 5 + 1))
    plt.savefig('./r2m_{}_{}.png'.format(policy, title), dpi = 1000)
    plt.close()

def main(out=None):
    facility = Layout(1, [2 for i in range(1)], [1 for i in range(2)], 1, 55)
    relocate_data, reject_data, block_data = get_res_ls(facility, 1/120, 4, 4, 293, 20)
    policy = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2', 'C3', 'C4']
    if out == 'r2m':
        for n in range(len(relocate_data[0])):
            policy_relocate = [i[n] for i in relocate_data]
            policy_reject = [i[n] for i in reject_data]
            policy_block = [i[n] for i in block_data]
            current_policy = policy[n]
            for p, q in zip([policy_relocate, policy_reject, policy_block], ['重定位次数', '拒绝次数', '阻挡次数']):
                regression2mean(p,q,current_policy)
    relocate_data = [np.mean([i[j] for i in relocate_data]) for j in range(len(relocate_data[0]))]
    reject_data = [np.mean([i[j] for i in reject_data]) for j in range(len(reject_data[0]))]
    block_data = [np.mean([i[j] for i in block_data]) for j in range(len(block_data[0]))]
    for i, j in zip([relocate_data, reject_data, block_data], ['重定位次数', '拒绝次数', '阻挡次数']):
        plot_policy(i, j)

main()
