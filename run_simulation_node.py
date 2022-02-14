import sympy
import argparse
import numpy as np

import equations
import data
from gp_utils import run_gp
import pickle
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint_adjoint as odeint

class NODE(nn.Module):

    def __init__(self, obs_dim=2, nhidden=50):
        super(NODE, self).__init__()
        self.sig = nn.Tanh()
        self.fc1 = nn.Linear(obs_dim, nhidden)
        self.fc1_5 = nn.Linear(nhidden, nhidden)
        self.fc2 = nn.Linear(nhidden, obs_dim)

    def forward(self, t, z):
        out = self.fc1(z)
        out = self.sig(out)
        out = self.fc1_5(out)
        out = self.sig(out)
        out = self.fc2(out)
        return out


def run(ode_name, ode_param, x_id, freq, n_sample, noise_ratio, alg, seed, n_seed):
    np.random.seed(999)
    torch.random.manual_seed(999)
    print(freq)

    ode = equations.get_ode(ode_name, ode_param)
    T = ode.T
    init_low = ode.init_low
    init_high = ode.init_high

    noise_sigma = ode.std_base * noise_ratio

    dg = data.DataGenerator(ode, T, freq, n_sample, noise_sigma, init_low, init_high)
    yt = dg.generate_data()

    if alg == 'one-step':
        n_step = 2
        t = torch.tensor(dg.solver.t[:n_step])
        yt_list = []

        for i in range(yt.shape[0] - (n_step - 1)):
            yt_list.append(yt[i:(i + n_step), :, :])

        y = torch.tensor(np.concatenate(yt_list, axis=1), dtype=torch.float32)

    else:
        t = torch.tensor(dg.solver.t)
        y = torch.tensor(yt, dtype=torch.float32)

    scalar = 1. / y.std()
    y = y * scalar
    y0 = y[0, ...]

    node = NODE(obs_dim=y0.shape[-1])
    optimizer = optim.Adam(node.parameters(), lr=0.01)

    niters = 3000
    test_freq = 100

    for itr in range(1, niters + 1):
        optimizer.zero_grad()

        y_hat = odeint(node, y0, t, method='dopri5', adjoint_options={"norm": "seminorm"})
        loss = torch.mean((y_hat - y) ** 2)
        loss.backward()
        optimizer.step()

        if itr % test_freq == 0:
            with torch.no_grad():
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))

    if alg == 'one-step':
        t = torch.tensor(dg.solver.t)
        y = torch.tensor(yt, dtype=torch.float32)
        scalar = 1. / y.std()
        y = y * scalar

    with torch.no_grad():
        dx = node(t, y.to(y0.dtype)) / scalar

    dxdt_hat = dx.cpu().numpy()

    print('NODE Training: Done.')

    X_train = yt
    X_train = X_train.reshape(X_train.shape[0] * X_train.shape[1], X_train.shape[2])

    y_train = dxdt_hat[:, :, x_id].flatten()
    assert X_train.shape[0] == y_train.shape[0]

    if alg == 'full':
        path_base = 'results_node/{}/noise-{}/sample-{}/freq-{}/'.format(ode_name, noise_ratio, n_sample, freq)
    else:
        path_base = 'results_node_one_step/{}/noise-{}/sample-{}/freq-{}/'.format(ode_name, noise_ratio, n_sample, freq)

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
    parser.add_argument("--alg", help="name of the benchmark", type=str, default='full', choices=['full', 'one-step'])
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
