import seaborn as sns
import os
import pandas as pd
import matplotlib.pyplot as plt


def merge_data(folder = './data'):
    data_ls = os.listdir(folder)
    df_ls = []
    size_array, capacity, relocation, reject, block, area_diff = [[] for i in range(6)]
    for data in data_ls:
        path = folder + '/' + data
        size = data.split('.')[0].split('_')
        this_data = pd.read_csv(path)
        this_data.sort_values(by='res', inplace = True)
        df_ls.append(this_data)
        size_array.append(size)
        capacity.append(this_data['facility_capacity'].values[-1])
        relocation.append(this_data['relocate'].values[-1])
        reject.append(this_data['reject'].values[-1])
        block.append(this_data['block'].values[-1])
        area_diff.append(this_data['area_diff'].values[-1])
    out = pd.DataFrame([size_array, capacity, relocation, reject, block, area_diff])
    out = out.T
    out.columns = ['size', 'capacity', 'relocation', 'reject', 'block', 'area_diff']
    out['width'] = out['size'].apply(lambda x: x[0]).astype(int)
    out['height'] = out['size'].apply(lambda x: x[1]).astype(int)
    out.sort_values(by='width', inplace = True)
    return out


def plot_layout(data, col, out_folder = 'pic'):
    sns.set(color_codes=True)
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    sns.lineplot(x = data['width'], y = data[col].astype(float))
    fig = plt.gcf()
    fig.savefig(out_folder + '/' + '{}.png'.format(col))
    fig.clear()

def plot_layout_lm(data, col, out_folder = 'pic'):
    sns.set(color_codes=True)
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    data[col] = data[col].astype(float)
    data['width'] = data['width'].astype(int)
    sns.lmplot(x = 'width', y = col,data = data, order=3)
    fig = plt.gcf()
    fig.savefig(out_folder + '/' + '{}.png'.format(col))
    fig.clear()

def main(folder = './data', mode = 'lm'):
    out = merge_data(folder)
    for i in out.columns[1: -2]:
        if mode == 'lm':
            plot_layout_lm(out, i)
        elif mode == 'line':
            plot_layout(out, i)

main(folder = './data_run12_A2', mode = 'lm')
