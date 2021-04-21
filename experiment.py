from ga_binary import *

total_area = 4800
demand = 400

for t in range(240, total_area // 20 + 1, 20):
    width = t
    height = total_area // t
    size = (width, height)
    var_num = (size[0] - 5) // (10 + 5)
    capacity = int(np.floor((size[1] - 3) / 5))
    bestsol, bestres, best_area_diff, best_relocate, best_reject, best_block, best_capacity, data = pipeline(lb = 0, ub = 7, var_num = var_num, capacity = capacity,
                    population = 50, mu = 0.05, pc = 0.5,
                    maxit = 30, arrive = 1 / 120, dwell = 4, demand = demand, size = size, times = 20)

    data.to_csv('./data/{}_{}.csv'.format(size[0], size[1]), index=None)