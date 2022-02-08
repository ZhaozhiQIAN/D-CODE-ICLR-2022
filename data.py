import numpy as np
import equations
import pickle

class DataGenerator:
    def __init__(self, ode, T, freq, n_sample, noise_sigma, init_low=0., init_high=1., return_list=False):
        self.ode = ode
        self.T = T
        self.freq = freq
        self.noise_sigma = noise_sigma
        self.solver = equations.ODESolver(ode, T, freq, return_list=return_list)
        self.return_list = return_list

        if isinstance(init_high, float) or isinstance(init_high, int):
            self.init_cond = np.random.uniform(init_low, init_high, (n_sample, ode.dim_x))
        else:
            # list of numbers
            assert len(init_high) == ode.dim_x
            tmp = list()
            for high, low in zip(init_high, init_low):
                tmp.append(np.random.uniform(low, high, (n_sample, 1)))
            self.init_cond = np.concatenate(tmp, axis=-1)

        self.xt = self.solver.solve(self.init_cond)
        if not self.return_list:
            self.eps = np.random.randn(*self.xt.shape) * noise_sigma
            self.yt = self.xt + self.eps

    def generate_data(self):
        return self.yt

class DataGeneratorReal:
    def __init__(self, dim_x, n_train):

        with open('data/real_data_c1.pkl', 'rb') as f:
            y_total = pickle.load(f)

        with open('data/real_data_mask_c1.pkl', 'rb') as f:
            mask = pickle.load(f)

        if dim_x == 1:
            self.yt = y_total[:, :, 0:1]
        else:
            self.yt = y_total

        self.mask = mask

        self.xt = self.yt.copy()

        self.yt_train = self.yt[:, :n_train, :].copy()
        self.yt_test = self.yt[:, n_train:, :].copy()

        self.mask_test = self.mask[:, n_train:].copy()

        # self.T = y_total.shape[0] - 1
        # self.solver = equations.ODESolver(equations.RealODEPlaceHolder(), self.T, 1.)

        self.T = 1.
        self.solver = equations.ODESolver(equations.RealODEPlaceHolder(), self.T, 364)
        self.noise_sigma = 0.001
        self.freq = 364

    def generate_data(self):
        return self.yt_train
