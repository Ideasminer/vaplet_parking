import cv2
import numpy as np
import os


def get_best(folder):
    path = os.listdir(folder)
    data = [[] for i in range(12)]
    count = 0
    for i in path:
        width = i.split('.')[0].split('_')[0]
        height = i.split('.')[0].split('_')[1]
        data[count].append(width)
        data[count].append(height)
        this_path = folder + '/' + i
        this_data = get_bestsolution(this_path)
        data[count].extend(this_data)
        count += 1
    data = pd.DataFrame(data)
    # data = data.T
    data.columns = ['width', 'height', 'sol', 'res', 'area_diff',
        'relocate', 'reject', 'block', 'facility_capacity']
    data = data[['width', 'height','sol', 'res', 'area_diff',
            'relocate', 'reject', 'block', 'facility_capacity']]
    return data

def get_layout(layout, num_stack):
    # 这里的layout为str类型的编码
    isl = [eval(s) for s in layout.split('_')[0]]
    gap = [eval(t) for t in layout.split('_')[1]]
    base_size = (2.5, 5)
    width = num_stack * base_size[0]
    whole_width = width + 3
    area_ls = []
    pos = [0, 0]
    for i in range(len(isl) + len(gap)):
        if i % 2 == 0:
            j = i // 2
            tpark = (gap[j] * base_size[0], width)
            cpos_lt = (pos[0], pos[1] + tpark[1])
            cpos_rb = (pos[0] + tpark[0], pos[1])
            cpos = [cpos_lt, cpos_rb]
            area_ls.append(cpos)
            pos = [pos[0] + tpark[0], pos[1]]
            pass_way_lt = (pos[0], pos[1] + tpark[1])
            pass_way_rb = (pos[0] + base_size[0], pos[1])
            pass_way = [pass_way_lt, pass_way_rb]
            area_ls.append(pass_way)
            pos = [pos[0] + base_size[0], pos[1]]
        if i % 2 == 1:
            j = i // 2
            isl_size = isl[j] * 2
            cpos_lt = [(pos[0] + i * base_size[1], pos[1] + j * base_size[0]) for j in range(1, num_stack + 1) for i in range(0, isl_size)]
            cpos_rb = [(pos[0] + i * base_size[1], pos[1] + j * base_size[0]) for j in range(0, num_stack) for i in range(1, isl_size + 1)]
            cpos = list(zip(cpos_lt, cpos_rb))
            cpos = [list(i) for i in cpos]
            area_ls.extend(cpos)
            pos = [pos[0] + isl_size * base_size[1], pos[1]]
    clear_way = [(0, whole_width), (pos[0], 0)]
    area_ls.append(clear_way)
    return area_ls

def plot(area_ls, solution = 2, title = None):
    if solution % 2 != 0:
        solution = 2 * solution // 2
    width = area_ls[-1][0][1]
    length = area_ls[-1][1][0]
    blank = 2
    img=np.zeros((int((np.ceil(width) + blank) * solution),int((np.ceil(length) + blank) * solution), 3), np.uint8)
    img.fill(255)
    for area in area_ls:
        lt = (int(solution * (area[0][0] + blank // 2)), int(solution * (width - area[0][1] + blank // 2)))
        rb = (int(solution * (area[1][0] + blank // 2)), int(solution * (width - area[1][1] + blank // 2)))
        cv2.rectangle(img,lt,rb,(0,0,0),2)
    # cv2.imshow("picture",img)
    cv2.waitKey(0)
    if title is None:
        title = 'test'
    cv2.imwrite('./{}.png'.format(title), img)

best_sol = get_best('./data_run9')
for sol in best_sol['sol'].values:
    area_ls = get_layout(sol, 10)
    plot(area_ls, 10, sol)
