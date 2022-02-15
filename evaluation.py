import argparse
import functools

import numpy as np

import equations
import data
from scipy.stats import ks_2samp
import pickle

import sys
import os

def run(ode_name, ode_param, x_id, freq, n_sample, noise_ratio, alg, seed, n_seed, eval_state):
    np.random.seed(999)

    seed_s = seed
    seed_e = n_seed

    if alg == 'diff':
        path_base = 'results/{}/noise-{}/sample-{}/freq-{}/'.format(ode_name, noise_ratio, n_sample, freq)
    elif alg == 'vi':
        path_base = 'results_vi/{}/noise-{}/sample-{}/freq-{}/'.format(ode_name, noise_ratio, n_sample, freq)
    elif alg == 'spline':
        path_base = 'results_spline/{}/noise-{}/sample-{}/freq-{}/'.format(ode_name, noise_ratio, n_sample, freq)
    elif alg == 'gp':
        path_base = 'results_gp/{}/noise-{}/sample-{}/freq-{}/'.format(ode_name, noise_ratio, n_sample, freq)
    elif alg == 'node':
        path_base = 'results_node/{}/noise-{}/sample-{}/freq-{}/'.format(ode_name, noise_ratio, n_sample, freq)
    else:
        path_base = 'results_node_one_step/{}/noise-{}/sample-{}/freq-{}/'.format(ode_name, noise_ratio, n_sample, freq)

    res_list = []
    for s in range(seed_s, seed_e):
        if x_id == 0:
            path = path_base + 'grad_seed_{}.pkl'.format(s)
        else:
            path = path_base + 'grad_x_{}_seed_{}.pkl'.format(x_id, s)

        try:
            with open(path, 'rb') as f:
                res = pickle.load(f)
            res_list.append(res)
        except Exception:
            pass

    correct_list = [res['correct'] for res in res_list]
    p_correct = np.mean(correct_list)
    std_correct = np.sqrt(p_correct * (1 - p_correct) / len(correct_list))

    # loop over res_list
    if eval_state:
        s_list = list()

        for res in res_list:
            try:
                ode_true = res['ode']
                f_hat = res['model'].execute
                dg_true = data.DataGenerator(ode_true, ode_true.T, freq=10, n_sample=100, noise_sigma=0., init_high=ode_true.init_high, init_low=ode_true.init_low)
                xt_true = dg_true.xt
                # ..., D
                xt_true = xt_true.reshape(xt_true.shape[0] * xt_true.shape[1], xt_true.shape[2])

                dxdt_hat = f_hat(xt_true).flatten()

                x_in = [xt_true[:, i] for i in range(xt_true.shape[1])]
                dxdt_true = ode_true._dx_dt(*x_in)[x_id]
                rmse = np.sqrt(np.mean((dxdt_hat - dxdt_true) ** 2))


                # if ode_true.dim_x == 1:
                #
                #     f_hat = res['model'].execute
                #     ode_hat = equations.InferredODE(ode_true.dim_x, f_hat_list=[f_hat], T=ode_true.T)
                #
                #     dg_true = data.DataGenerator(ode_true, ode_true.T, freq=10, n_sample=100, noise_sigma=0.,
                #                                  init_high=ode_true.init_high)
                #
                #     dg_hat = data.DataGenerator(ode_hat, ode_true.T, freq=10, n_sample=100, noise_sigma=0.,
                #                                 init_high=ode_true.init_high)
                #
                #     xt_hat = dg_hat.yt.flatten()
                #     xt_true = dg_true.yt.flatten()
                # else:
                #     dg_true = data.DataGenerator(ode_true, ode_true.T, freq=10, n_sample=1000, noise_sigma=0.,
                #                                  init_high=ode_true.init_high)
                #     xt_true = dg_true.xt
                #     xt_true = xt_true.reshape(xt_true.shape[0] * xt_true.shape[1], xt_true.shape[2]).T
                #
                #     def f(x, ind):
                #         ret = ode_true._dx_dt(*x[0, :])
                #         return ret[ind]
                #
                #     f_hat_list = []
                #     for i in range(ode_true.dim_x):
                #         if i == x_id:
                #             f_hat_list.append(res['model'].execute)
                #         else:
                #             f_hat_list.append(functools.partial(f, ind=i))
                #
                #     ode_hat = equations.InferredODE(ode_true.dim_x, f_hat_list=f_hat_list, T=ode_true.T)
                #     dg_hat = data.DataGenerator(ode_hat, ode_true.T, freq=10, n_sample=1000, noise_sigma=0.,
                #                                 init_high=ode_true.init_high)
                #
                #     xt_hat = dg_hat.xt
                #     xt_hat = xt_hat.reshape(xt_hat.shape[0] * xt_hat.shape[1], xt_hat.shape[2]).T

                # KS statistics: smaller better - state space divergence
                # s = ks_2samp(xt_hat, xt_true).statistic
                s_list.append(rmse)
            except ValueError:
                pass

        s_mean = np.mean(s_list)
        s_std = np.std(s_list) / np.sqrt(len(res_list))
    else:
        s_mean = 0
        s_std = 0

    # ode_name, ode_param, x_id, freq, n_sample, noise_ratio, alg

    print_list = [ode_name, freq, n_sample, noise_ratio, alg, p_correct, std_correct, s_mean, s_std]
    print_list = [str(x) for x in print_list]
    if not np.isnan(p_correct):
        print(','.join(print_list))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ode_name", help="name of the ode", type=str)
    parser.add_argument("--ode_param", help="parameters of the ode (default: None)", type=str, default=None)
    parser.add_argument("--x_id", help="ID of the equation to be learned", type=int, default=0)
    parser.add_argument("--freq", help="sampling frequency", type=float, default=10)
    parser.add_argument("--n_sample", help="number of trajectories", type=int, default=100)
    parser.add_argument("--noise_sigma", help="noise level (default 0)", type=float, default=0.)
    parser.add_argument("--alg", help="name of the benchmark", type=str, default='diff', choices=['diff', 'vi', 'node', 'spline', 'gp', 'node_one_step'])
    parser.add_argument("--seed", help="random seed", type=int, default=0)
    parser.add_argument("--n_seed", help="random seed", type=int, default=100)
    parser.add_argument("--eval_state", help="If evaluate state distri.", type=bool, default=True)

    args = parser.parse_args()

    if args.ode_param is not None:
        param = [float(x) for x in args.ode_param.split(',')]
    else:
        param = None
    if args.freq >= 1:
        freq = int(args.freq)
    else:
        freq = args.freq
    run(args.ode_name, param, args.x_id, freq, args.n_sample,
        args.noise_sigma, args.alg, seed=args.seed, n_seed=args.n_seed, eval_state=args.eval_state)
