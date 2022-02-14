import sympy
import argparse
import numpy as np

import equations
import data
# from dsr_utils import run_dsr
from gp_utils import run_gp
from interpolate import num_diff, num_diff_gp
import pickle
import os
import time

# # set up ODE config
# ode_param = None
# x_id = 0
#
# # data generation config
# freq = 10
# n_sample = 100
# noise_sigma = 0.0
#
# # set up algorithm config
# alg = 'gp'

def run(ode_name, ode_param, x_id, freq, n_sample, noise_ratio, alg, seed, n_seed):
    np.random.seed(999)
    print(freq)

    ode = equations.get_ode(ode_name, ode_param)
    T = ode.T
    init_low = ode.init_low
    init_high = ode.init_high
    has_coef = ode.has_coef

    noise_sigma = ode.std_base * noise_ratio

    dg = data.DataGenerator(ode, T, freq, n_sample, noise_sigma, init_low, init_high)
    yt = dg.generate_data()

    if noise_sigma == 0:
        dxdt_hat = (yt[1:, :, :] - yt[:-1, :, :]) / (dg.solver.t[1:] - dg.solver.t[:-1])[:, None, None]
    elif alg != 'gp':
        dxdt_hat = num_diff(yt, dg, alg)
    else:
        dxdt_hat, xt_hat = num_diff_gp(yt, dg, ode)

    print('Numerical differentiation: Done.')

    # if alg != 'gp':
    X_train = yt[:-1, :, :]
    # else:
    #     X_train = xt_hat[:-1, :, :]
    X_train = X_train.reshape(X_train.shape[0] * X_train.shape[1], X_train.shape[2])

    y_train = dxdt_hat[:, :, x_id].flatten()
    assert X_train.shape[0] == y_train.shape[0]

    if alg == 'tv':
        path_base = 'results/{}/noise-{}/sample-{}/freq-{}/'.format(ode_name, noise_ratio, n_sample, freq)
    elif alg == 'gp':
        path_base = 'results_gp/{}/noise-{}/sample-{}/freq-{}/'.format(ode_name, noise_ratio, n_sample, freq)
    else:
        path_base = 'results_spline/{}/noise-{}/sample-{}/freq-{}/'.format(ode_name, noise_ratio, n_sample, freq)

    if not os.path.isdir(path_base):
        os.makedirs(path_base)

    for s in range(seed, seed+n_seed):
        print(' ')
        print('Running with seed {}'.format(s))
        if x_id == 0:
            path = path_base + 'grad_seed_{}.pkl'.format(s)
        else:
            path = path_base + 'grad_x_{}_seed_{}.pkl'.format(x_id, s)

        if os.path.isfile(path):
            print('Skipping seed {}'.format(s))
            continue
        start = time.time()
        f_hat, est_gp = run_gp(X_train, y_train, ode, x_id, s)
        print(f_hat)
        f_true = ode.get_expression()[x_id]
        if not isinstance(f_true, tuple):
            correct = sympy.simplify(f_hat - f_true) == 0
        else:
            correct_list = [sympy.simplify(f_hat - f) == 0 for f in f_true]
            correct = max(correct_list) == 1

        end = time.time()

        with open(path, 'wb') as f:
            pickle.dump({
                'model': est_gp._program,
                'X_train': X_train,
                'y_train': y_train,
                'seed': s,
                'correct': correct,
                'f_hat': f_hat,
                'ode': ode,
                'noise_ratio': noise_ratio,
                'noise_sigma': noise_sigma,
                'dg': dg,
                'time': end-start,
            }, f)

        print(f_hat)
        print(correct)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ode_name", help="name of the ode", type=str)
    parser.add_argument("--ode_param", help="parameters of the ode (default: None)", type=str, default=None)
    parser.add_argument("--x_id", help="ID of the equation to be learned", type=int, default=0)
    parser.add_argument("--freq", help="sampling frequency", type=float, default=10)
    parser.add_argument("--n_sample", help="number of trajectories", type=int, default=100)
    parser.add_argument("--noise_sigma", help="noise level (default 0)", type=float, default=0.)
    parser.add_argument("--alg", help="name of the benchmark", type=str, default='tv', choices=['tv', 'spline', 'gp'])
    parser.add_argument("--seed", help="random seed", type=int, default=0)
    parser.add_argument("--n_seed", help="random seed", type=int, default=10)

    args = parser.parse_args()
    print('Running with: ', args)

    if args.ode_param is not None:
        param = [float(x) for x in args.ode_param.split(',')]
    else:
        param = None
    if args.freq >= 1:
        freq = int(args.freq)
    else:
        freq = args.freq
    run(args.ode_name, param, args.x_id, freq, args.n_sample, args.noise_sigma, args.alg, seed=args.seed, n_seed=args.n_seed)
