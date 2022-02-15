import sympy
import argparse
import numpy as np

import equations
import data
import pickle

import sys
import os
from gplearn.genetic import SymbolicRegressor


def get_grid():
    n_population_size = [15000]
    p_crossover = [0.6, 0.7, 0.8]
    p_subtree_mutation = [0.05, 0.1, 0.15]
    p_hoist_mutation = [0.02, 0.05, 0.07]
    p_point_mutation = [0.05, 0.1, 0.15]
    p_add = [1, 3, 5]
    p_sub = [1, 3, 5]
    p_mul = [1, 3, 5, 10]
    # p_div = [1, 3, 5]

    # +, -, *, log
    function_set = dict()
    function_set['add'] = np.random.choice(p_add)
    # function_set['sub'] = np.random.choice(p_sub)
    function_set['mul'] = np.random.choice(p_mul)
    # function_set['div'] = np.random.choice(p_div)
    function_set['log'] = 1
    # function_set['neg'] = 1
    # function_set['cos'] = 1

    if np.random.uniform() < 0.5:
        function_set = {
            # 'add': 25, 'sub': 25, 'mul': 50, 'div': 5, 'log': 5, 'cos': 5
            'add': 25, 'mul': 50, 'log': 5
            # 'add': 25, 'mul': 50, 'log': 5
        }

    p_arr = np.array([
        np.random.choice(p_crossover),
        np.random.choice(p_subtree_mutation),
        np.random.choice(p_hoist_mutation),
        np.random.choice(p_point_mutation)
    ])
    p_arr = p_arr / p_arr.sum() * 0.95

    random_grid = {'population_size': np.random.choice(n_population_size),
                   'p_crossover': p_arr[0],
                   'p_subtree_mutation': p_arr[1],
                   'p_hoist_mutation': p_arr[2],
                   'p_point_mutation': p_arr[3],
                   'function_set': function_set}
    return random_grid


def run(ode_name, ode_param, x_id, freq, n_sample, noise_sigma, alg, itr, const_max, const_min, seed):
    assert noise_sigma == 0
    assert alg == 'gp'
    np.random.seed(999)

    ode = equations.get_ode(ode_name, ode_param)
    T = ode.T
    init_low = 0
    init_high = ode.init_high
    has_coef = ode.has_coef

    dg = data.DataGenerator(ode, T, freq, n_sample, noise_sigma, init_low, init_high)
    yt = dg.generate_data()

    dxdt_hat = (yt[1:, :, :] - yt[:-1, :, :]) / (dg.solver.t[1:] - dg.solver.t[:-1])[:, None, None]

    X_train = yt[:-1, :, :]
    X_train = X_train.reshape(X_train.shape[0] * X_train.shape[1], X_train.shape[2])

    y_train = dxdt_hat[:, :, x_id].flatten()
    assert X_train.shape[0] == y_train.shape[0]

    #training loop
    loss_list = list()
    param_list = list()
    model_list = list()

    np.random.seed(seed)

    for i in range(itr):
        grid = get_grid()
        print(grid)
        est_gp = SymbolicRegressor(generations=20, stopping_criteria=0.01,
                                   max_samples=0.9, verbose=0,
                                   parsimony_coefficient=0.01, random_state=seed,
                                   init_depth=(1, 6), n_jobs=2,
                                   const_range=(const_min, const_max), low_memory=True, **grid)
        est_gp.fit(X_train, y_train)
        loss = est_gp.run_details_['best_oob_fitness'][-1]
        print(loss)
        loss_list.append(loss)
        param_list.append(grid)
        model_list.append(est_gp._program)

    best_param_ind = loss_list.index(min(loss_list))
    best_param = param_list[best_param_ind]
    print(best_param)

    if x_id == 0:
        path = 'param/{}_seed_{}_gp.pkl'.format(ode_name, seed)
    else:
        path = 'param/{}_x_{}_seed_{}_gp.pkl'.format(ode_name, x_id, seed)

    with open(path, 'wb') as f:
        pickle.dump({
            'loss_list': loss_list,
            'param_list': param_list,
            'model_list': model_list,
            'best_param': best_param,
            'X_train': X_train,
            'y_train': y_train
        }, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ode_name", help="name of the ode", type=str)
    parser.add_argument("--ode_param", help="parameters of the ode (default: None)", type=str, default=None)
    parser.add_argument("--x_id", help="ID of the equation to be learned", type=int, default=0)
    parser.add_argument("--freq", help="sampling frequency", type=int, default=10)
    parser.add_argument("--n_sample", help="number of trajectories", type=int, default=100)
    parser.add_argument("--noise_sigma", help="noise level (default 0)", type=float, default=0.)
    parser.add_argument("--itr", help="number of search iteration (default 10)", type=int, default=10)
    parser.add_argument("--alg", help="name of the benchmark", type=str, default='gp', choices=['gp'])
    parser.add_argument("--seed", help="random seed", type=int, default=999)
    parser.add_argument("--const_max", help="max_constant", type=float, default=5)
    parser.add_argument("--const_min", help="min_constant", type=float, default=5)

    args = parser.parse_args()
    print('Running with: ', args)

    if args.ode_param is not None:
        param = [float(x) for x in args.ode_param.split(',')]
    else:
        param = None

    run(args.ode_name, param, args.x_id, args.freq, args.n_sample, args.noise_sigma, args.alg, args.itr, args.const_max, args.const_min, seed=args.seed)
