import sympy
import argparse
import numpy as np

import equations
import data
from gp_utils import run_gp_ode
from interpolate import get_ode_data
import pickle
import os
import time
import basis


def run(ode_name, ode_param, x_id, freq, n_sample, noise_ratio, seed, n_seed, n_basis, basis_str):
    np.random.seed(999)

    ode = equations.get_ode(ode_name, ode_param)
    T = ode.T
    init_low = 0
    init_high = ode.init_high

    if basis_str == 'sine':
        basis_obj = basis.FourierBasis
    else:
        basis_obj = basis.CubicSplineBasis

    noise_sigma = ode.std_base * noise_ratio

    dg = data.DataGenerator(ode, T, freq, n_sample, noise_sigma, init_low, init_high)
    yt = dg.generate_data()

    ode_data, X_ph, y_ph, t_new = get_ode_data(yt, x_id, dg, ode, n_basis, basis_obj)

    path_base = 'results_vi/{}/noise-{}/sample-{}/freq-{}/n_basis-{}/basis-{}'.\
        format(ode_name, noise_ratio, n_sample, freq, n_basis, basis_str)

    if not os.path.isdir(path_base):
        os.makedirs(path_base)

    for s in range(seed, seed+n_seed):
        print(' ')
        print('Running with seed {}'.format(s))
        start = time.time()

        f_hat, est_gp = run_gp_ode(ode_data, X_ph, y_ph, ode, x_id, s)

        f_true = ode.get_expression()[x_id]
        if not isinstance(f_true, tuple):
            correct = sympy.simplify(f_hat - f_true) == 0
        else:
            correct_list = [sympy.simplify(f_hat - f) == 0 for f in f_true]
            correct = max(correct_list) == 1

        if x_id == 0:
            path = path_base + 'grad_seed_{}.pkl'.format(s)
        else:
            path = path_base + 'grad_x_{}_seed_{}.pkl'.format(x_id, s)
        end = time.time()

        with open(path, 'wb') as f:
            pickle.dump({
                'model': est_gp._program,
                'ode_data': ode_data,
                'seed': s,
                'correct': correct,
                'f_hat': f_hat,
                'ode': ode,
                'noise_ratio': noise_ratio,
                'noise_sigma': noise_sigma,
                'dg': dg,
                't_new': t_new,
                'time': end - start,
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
    parser.add_argument("--n_basis", help="number of basis function", type=int, default=50)
    parser.add_argument("--basis", help="basis function", type=str, default='sine')

    parser.add_argument("--seed", help="random seed", type=int, default=0)
    parser.add_argument("--n_seed", help="random seed", type=int, default=10)

    args = parser.parse_args()
    print('Running with: ', args)

    if args.ode_param is not None:
        param = [float(x) for x in args.ode_param.split(',')]
    else:
        param = None

    run(args.ode_name, param, args.x_id, args.freq, args.n_sample, args.noise_sigma, seed=args.seed, n_seed=args.n_seed,
        n_basis=args.n_basis, basis_str=args.basis)
