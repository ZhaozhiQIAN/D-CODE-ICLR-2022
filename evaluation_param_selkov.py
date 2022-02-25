import argparse
import functools

import numpy as np

import equations
import data
from scipy.stats import ks_2samp
import pickle
from gp_utils import *

import sys
import os

def gp_post(f_star, ode, tol=None, expand=False):
    VarDict = ode.get_var_dict()
    f_star_list, var_list, coef_list = parse_program_to_list(f_star.program)
    f_star_infix = generator.Generator.prefix_to_infix(f_star_list, variables=var_list, coefficients=coef_list)
    f_star_infix2 = f_star_infix.replace('{', '').replace('}', '')
    if f_star_infix2 == f_star_infix:
        f_star_sympy = generator.Generator.infix_to_sympy(f_star_infix, VarDict, "simplify")
        return f_star_sympy

    f_star_sympy = generator.Generator.infix_to_sympy(f_star_infix2, VarDict, "simplify")

    return f_star_sympy

def std_RMSE(err_sq):
    rmse_list = []
    for i in range(500):
        new_err = err_sq[np.random.randint(0, len(err_sq), err_sq.shape)]
        rmse_itr = np.sqrt(np.mean(new_err))
        rmse_list.append(rmse_itr)
    return np.std(np.array(rmse_list))

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
    else:
        path_base = 'results_node/{}/noise-{}/sample-{}/freq-{}/'.format(ode_name, noise_ratio, n_sample, freq)

    res_list = []
    sigma_hat_list = []
    rho_hat_list = []

    for s in range(seed_s, seed_e):
        if x_id == 0:
            path = path_base + 'grad_seed_{}.pkl'.format(s)
        else:
            path = path_base + 'grad_x_{}_seed_{}.pkl'.format(x_id, s)

        try:
            with open(path, 'rb') as f:
                res = pickle.load(f)

            if res['correct']:
                f_hat = gp_post(res['model'], res['ode'])
                VarDict = res['ode'].get_var_dict()
                if x_id == 0:
                    sigma_hat = -1 * sympy.Poly(f_hat).coeff_monomial(VarDict['X0'])
                    rho_hat = sympy.Poly(f_hat).coeff_monomial(1)
                else:
                    sigma_hat = sympy.Poly(f_hat).coeff_monomial(VarDict['X0'])
                    rho_hat = 0
                sigma_hat_list.append(float(sigma_hat))
                rho_hat_list.append(float(rho_hat))


            res_list.append(res)
        except Exception:
            pass

    sigma_hat_arr = np.array(sigma_hat_list)
    rho_hat_arr = np.array(rho_hat_list)
    sigma_err = np.sqrt(np.mean((sigma_hat_arr - 0.1) ** 2))
    sigma_std = std_RMSE((sigma_hat_arr - 0.1) ** 2)


    rho_err = np.sqrt(np.mean((rho_hat_arr - 0.75)**2))
    rho_std = std_RMSE((rho_hat_arr - 0.1) ** 2)

    # ode_name, ode_param, x_id, freq, n_sample, noise_ratio, alg

    print_list = [ode_name, freq, n_sample, noise_ratio, alg, sigma_err, sigma_std, rho_err, rho_std]
    print_list = [str(x) for x in print_list]

    print(','.join(print_list))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ode_name", help="name of the ode", type=str)
    parser.add_argument("--ode_param", help="parameters of the ode (default: None)", type=str, default=None)
    parser.add_argument("--x_id", help="ID of the equation to be learned", type=int, default=0)
    parser.add_argument("--freq", help="sampling frequency", type=int, default=10)
    parser.add_argument("--n_sample", help="number of trajectories", type=int, default=100)
    parser.add_argument("--noise_sigma", help="noise level (default 0)", type=float, default=0.)
    parser.add_argument("--alg", help="name of the benchmark", type=str, default='diff', choices=['diff', 'vi', 'node', 'spline', 'gp'])
    parser.add_argument("--seed", help="random seed", type=int, default=0)
    parser.add_argument("--n_seed", help="random seed", type=int, default=100)
    parser.add_argument("--eval_state", help="If evaluate state distri.", type=bool, default=False)

    args = parser.parse_args()

    if args.ode_param is not None:
        param = [float(x) for x in args.ode_param.split(',')]
    else:
        param = None

    run(args.ode_name, param, args.x_id, args.freq, args.n_sample,
        args.noise_sigma, args.alg, seed=args.seed, n_seed=args.n_seed, eval_state=args.eval_state)
