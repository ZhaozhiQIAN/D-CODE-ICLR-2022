import argparse
import numpy as np

import data
import equations
from gp_utils import run_gp_ode
from interpolate import get_ode_data
import pickle
import os
import time


def run(dim_x, x_id, n_sample, seed, n_seed):
    np.random.seed(999)
    ode_name = 'real'

    dg = data.DataGeneratorReal(dim_x, n_sample)

    yt = dg.generate_data()
    ode = equations.RealODEPlaceHolder()
    ode_data, X_ph, y_ph, t_new = get_ode_data(yt, x_id, dg, ode)

    path_base = 'results_vi/{}/sample-{}/dim-{}/'.format(ode_name, n_sample, dim_x)

    if not os.path.isdir(path_base):
        os.makedirs(path_base)

    for s in range(seed, seed+n_seed):
        print(' ')
        print('Running with seed {}'.format(s))
        start = time.time()
        f_hat, est_gp = run_gp_ode(ode_data, X_ph, y_ph, ode, x_id, s)

        print(f_hat)

        if x_id == 0:
            path = path_base + 'grad_seed_{}.pkl'.format(s)
        else:
            path = path_base + 'grad_x_{}_seed_{}.pkl'.format(x_id, s)
        end = time.time()

        with open(path, 'wb') as f:
            pickle.dump({
                'model': est_gp._program,
                'gp': est_gp,
                'ode_data': ode_data,
                'seed': s,
                'f_hat': f_hat,
                'ode': ode,
                'dg': dg,
                't_new': t_new,
                'time': end - start,
            }, f)

        print(f_hat)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim_x", help="number of dimensions", type=int, default=2)
    parser.add_argument("--x_id", help="ID of the equation to be learned", type=int, default=0)
    parser.add_argument("--n_sample", help="number of trajectories", type=int, default=100)
    parser.add_argument("--seed", help="random seed", type=int, default=0)
    parser.add_argument("--n_seed", help="random seed", type=int, default=10)

    args = parser.parse_args()
    print('Running with: ', args)

    run(args.dim_x, args.x_id, args.n_sample, seed=args.seed, n_seed=args.n_seed)
