import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
plt.rcParams['font.sans-serif'] = ['FangSong']
plt.rcParams['axes.unicode_minus'] = False

def regression(x, y):
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    poly = PolynomialFeatures(degree=6)
    x = poly.fit_transform(x)
    model = LinearRegression()
    model.fit(x, y)
    return model

def smooth(data, x, y, col, scale = 4):
    X = []
    Y = []
    Out_Z = []
    for v in range(len(x)):
        if v != len(x) - 1:
            this_x = x[v]
            next_x = x[v + 1]
            out_x = np.linspace(this_x, next_x, scale).tolist()[:-1]
            X.extend(out_x)
        else:
            this_x = x[v]
            X.append(this_x)
            break
    for v in range(len(y)):
        if v != len(y) - 1:
            this_y = y[v]
            next_y = y[v + 1]
            out_y = np.linspace(this_y, next_y, scale).tolist()[:-1]
            Y.extend(out_y)
        else:
            this_y = y[v]
            Y.append(this_y)
            break
    X = np.array(X)
    Y = np.array(Y)
    X, Y = np.meshgrid(X,Y)
    shape = X.shape
    relocate = data[col].astype(int).values.reshape(8, 12)
    for i in range(relocate.shape[0]):
        Z = []
        for j in range(relocate.shape[1]):
            if j != relocate.shape[1] - 1:
                this_z = relocate[i][j]
                next_z = relocate[i][j + 1]
                out_z = np.linspace(this_z, next_z, scale).tolist()[:-1]
                Z.extend(out_z)
            else:
                this_z = relocate[i][j]
                Z.append(this_z)
                Out_Z.append(Z)
                break
    Out_Z = np.array(Out_Z)
    temp_Z = []
    for j in range(Out_Z.shape[1]):
        Z = []
        for i in range(Out_Z.shape[0]):
            if i != Out_Z.shape[0] - 1:
                this_z = Out_Z[i][j]
                next_z = Out_Z[i + 1][j]
                out_z = np.linspace(this_z, next_z, scale).tolist()[:-1]
                Z.extend(out_z)
            else:
                this_z = Out_Z[i][j]
                Z.append(this_z)
                temp_Z.append(Z)
                break
    Out_Z = np.transpose(np.array(temp_Z))
    return X, Y, Out_Z

def surf_plot(data, col, folder = './', scale = 4):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    fig.subplots_adjust(left=0.05, right=0.95, top = 0.90, bottom=0.05)
    X = data[data['策略'] == 'A1']['长宽比'].astype(float).values
    Y = np.array([i for i in range(1, 9)])
    x, y, z = smooth(data, X, Y, col, scale = scale)
    surf = ax.plot_surface(x,y,z, rstride = 1, cstride = 1,cmap=cm.jet,
                        linewidth=0, antialiased=False)
    ax.set_title('{}变化趋势图'.format(col), pad = 15)
    ax.set_xlabel('长宽比', labelpad = 8)
    ax.set_ylabel('策略', labelpad = 10)
    ax.set_zlabel(col, labelpad = 10)
    ax.set_yticks([i for i in range(1, 9)])
    ax.set_yticklabels([r'A1',r'A2',r'B1',r'B2',r'C1',r'C2',r'C3',r'C4'])
    zmin = np.min(z) // 5 * 5
    zmax = np.max(z) // 5 * 5
    if zmax == zmin:
        zmax = np.max(z)
        zmin = np.min(z)
    frac = int(np.ceil((zmax - zmin)/ 5))
    if frac != 0:
        ax.set_zticks([i for i in np.arange(zmin, zmax, frac)])
    else:
        frac = (zmax - zmin)/ 5
        ax.set_zticks([i for i in np.arange(zmin, zmax, frac)])
    ax.set_zlim(zmin, zmax)
    fig.colorbar(surf, shrink=0.5, aspect=15, pad = 0.08)
    plt.savefig(folder + '{}.png'.format(col), dpi = 1000, transparent = False)
    plt.close()

def curve_plot(data, col, folder = './', scale = 100):
    color_ls = ['red','blue','green','orange','black', 'violet','brown', 'yellow']
    policy = [r'A1',r'A2',r'B1',r'B2',r'C1',r'C2',r'C3',r'C4']
    marker = ['^','x','v','p','o','d','h','1']
    fig = plt.figure()
    ax = fig.gca()
    fig.subplots_adjust(left=0.1, right=0.95, top = 0.90, bottom=0.1)
    line = []
    scatter = []
    point = []
    y_upper = np.max(data[col].values)
    y_lower = np.min(data[col].values)
    x_upper = np.max(data['长宽比'].values)
    x_lower = np.min(data['长宽比'].values)
    for i in range(len(policy)):
        p = policy[i]
        c = color_ls[i]
        m = marker[i]
        cdata = data[data['策略'] == p]
        x = cdata['长宽比'].values
        y = cdata[col].values
        model = regression(x, y)
        expand_x = np.linspace(np.min(x), np.max(x), scale).reshape(-1, 1)
        expand_x_copy = expand_x.copy()
        poly = PolynomialFeatures(degree=6)
        expand_x = poly.fit_transform(expand_x)
        expand_y = model.predict(expand_x).reshape(-1, 1)
        line.append(ax.plot(expand_x_copy, expand_y, c=c))
        scatter.append(ax.scatter(x, y, c = c, marker = m))
        if i == 0:
            point.extend([expand_x_copy[scale // 2],expand_y[scale // 2]])
    arrow_point = tuple(point)
    text_point = (point[0] + (x_upper - x_lower) / 6, point[1] + (y_upper - y_lower) / 5)
    ax.legend(scatter, policy,loc = 1, title='策略')
    ax.annotate("趋势线", arrow_point, xycoords='data',xytext=text_point,
         arrowprops=dict(arrowstyle='->'))
    ax.set_title('{}变化趋势图'.format(col), pad = 15)
    ax.set_xlabel('长宽比', labelpad = 8)
    ax.set_ylabel(col, labelpad = 10)
    plt.savefig(folder + '{}.png'.format(col), dpi = 1000, transparent = False)
    plt.close()

def single_line_plot(data, policy, col, folder = './', scale = 100):
    if not isinstance(col, list):
        col = [col]
    counter = 0
    fig = plt.figure()
    ax = fig.gca()
    line_ls = []
    right = 0.95
    for c in col:
        if counter == 1:
            ax = ax.twinx()
            right = 0.85
        if c == '阻挡次数' or '面积差':
            fig.subplots_adjust(left=0.16, right=right, top = 0.90, bottom=0.1)
        else:
            fig.subplots_adjust(left=0.1, right=right, top = 0.90, bottom=0.1)
        point = []
        cdata = data[data['策略'] == policy]
        x = cdata['长宽比'].values
        y = cdata[c].values
        model = regression(x, y)
        expand_x = np.linspace(np.min(x), np.max(x), scale).reshape(-1, 1)
        expand_x_copy = expand_x.copy()
        poly = PolynomialFeatures(degree=6)
        expand_x = poly.fit_transform(expand_x)
        expand_y = model.predict(expand_x).reshape(-1, 1)
        if counter == 1:
            ax.scatter(x, y, c = 'green', marker = 'x')
            line = ax.plot(expand_x_copy, expand_y, c='green')
            line_ls.append(line)
            ax.legend(line, ['{}趋势线'.format(c)], loc=2)
        else:
            ax.scatter(x, y, c = 'red', marker = '^')
            line = ax.plot(expand_x_copy, expand_y, c='red')
            line_ls.append(line)
            ax.legend(line, ['{}趋势线'.format(c)], loc=1)
        point.extend([expand_x_copy[scale // 2],expand_y[scale // 2]])
        ax.set_ylabel(c, labelpad = 10)
        counter += 1
    if counter == 2:
        name = col[0] + '-' + col[1]
        ax.set_title('{}变化趋势图'.format(name), pad = 15)
    else:
        ax.set_title('{}变化趋势图'.format(c), pad = 15)
    ax.set_xlabel('长宽比', labelpad = 8)
    if counter == 2:
        name = col[0] + '-' + col[1]
        plt.savefig(folder + '{}_{}.png'.format(policy,name), dpi = 1000, transparent = False)
    else:
        plt.savefig(folder + '{}_{}.png'.format(policy,c), dpi = 1000, transparent = False)
    plt.close()

def main(mode = 'surf'):
    data = pd.read_excel(r'summary.xlsx')
    # col = ['重定位次数', '拒绝次数', '阻挡次数', '容量', '面积差']
    # col = ['单泊位面积']
    if mode == 'surf':
        for c in col:
            surf_plot(data, c, folder = './pic/3dres/', scale = 30)
    elif mode == 'curve':
        for c in col:
            curve_plot(data, c, folder = './pic/2dres/')
    elif mode == 'single':
        policy = [r'A1',r'A2',r'B1',r'B2',r'C1',r'C2',r'C3',r'C4']
        for p in policy:
            single_line_plot(data, p, ['重定位次数','单泊位面积'], folder = './pic/singleres/new/')

main(mode = 'single')
