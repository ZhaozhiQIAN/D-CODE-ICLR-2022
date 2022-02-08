import numpy as np
import abc
from scipy.interpolate import BSpline
from sklearn.linear_model import LassoCV
from scipy.linalg import lstsq
from tvregdiff.tvregdiff import TVRegDiff
import gppca
from config import get_interpolation_config
from integrate import generate_grid
from derivative import dxdt


def get_ode_data_noise_free(yt, x_id, dg, ode):
    t = dg.solver.t
    freq = dg.freq
    t_new = t

    weight = np.ones_like(t_new)
    weight[0] /= 2
    weight[-1] /= 2
    weight = weight / weight.sum() * dg.T

    X_sample = yt
    config = get_interpolation_config(ode, 0)
    n_basis = config['n_basis']
    basis = config['basis']

    basis_func = basis(dg.T, n_basis)

    g_dot = basis_func.design_matrix(t_new, derivative=True)
    g = basis_func.design_matrix(t_new, derivative=False)

    Xi = X_sample[:, :, x_id]

    c = (Xi * weight[:, None]).T @ g_dot
    ode_data = {
        'x_hat': X_sample,
        'g': g,
        'c': c,
        'weights': weight
    }
    X_ph = np.zeros((X_sample.shape[1], X_sample.shape[2]))
    y_ph = np.zeros(X_sample.shape[1])

    return ode_data, X_ph, y_ph, t_new


def get_ode_data(yt, x_id, dg, ode, config_n_basis=None, config_basis=None):
    t = dg.solver.t
    noise_sigma = dg.noise_sigma
    freq = dg.freq

    if noise_sigma == 0:
        return get_ode_data_noise_free(yt, x_id, dg, ode)

    X_sample_list = list()
    pca_list = []
    # for each dimension
    assert yt.shape[-1] > 0
    for d in range(yt.shape[-1]):
        config = get_interpolation_config(ode, d)
        Y = yt[:, :, d]
        r = config['r']
        if r < 0:
            r = Y.shape[1]

        if 'sigma_in_mul' in config.keys():
            sigma_in_mul = config['sigma_in_mul']
            sigma_in = sigma_in_mul / freq
        else:
            sigma_in = config['sigma_in']
        freq_int = config['freq_int']

        pca = gppca.GPPCA0(r, Y, t, noise_sigma, sigma_out=ode.std_base, sigma_in=sigma_in)

        t_new, weight = generate_grid(dg.T, freq_int)
        X_sample = pca.get_predictive(new_sample=1, t_new=t_new)
        X_sample = X_sample.reshape(len(t_new), X_sample.size // len(t_new), 1)
        X_sample_list.append(X_sample)
        pca_list.append(pca)

    X_sample = np.concatenate(X_sample_list, axis=-1)
    # check smaller than zero
    if ode.positive:
        X_sample[X_sample <= 1e-6] = 1e-6
    config = get_interpolation_config(ode, x_id)
    if config_n_basis is None:
        n_basis = config['n_basis']
    else:
        n_basis = config_n_basis
    if config_basis is None:
        basis = config['basis']
    else:
        basis = config_basis

    basis_func = basis(dg.T, n_basis)
    g = basis_func.design_matrix(t_new, derivative=False)

    # compute c using a much larger grid
    t_new_c, weight_c = generate_grid(dg.T, 1000)
    g_dot = basis_func.design_matrix(t_new_c, derivative=True)
    Xi = pca_list[x_id].get_predictive(new_sample=1, t_new=t_new_c)
    Xi = Xi.reshape(len(t_new_c), Xi.size // len(t_new_c))
    c = (Xi * weight_c[:, None]).T @ g_dot

    ode_data = {
        'x_hat': X_sample,
        'g': g,
        'c': c,
        'weights': weight
    }
    X_ph = np.zeros((X_sample.shape[1], X_sample.shape[2]))
    y_ph = np.zeros(X_sample.shape[1])

    return ode_data, X_ph, y_ph, t_new


def num_dff_tuning(yt, dg, alg):
    yt0 = dg.xt
    dxdt_hat = (yt0[1:, :, :] - yt0[:-1, :, :]) / (dg.solver.t[1:] - dg.solver.t[:-1])[:, None, None]

    y_single = yt[:, 0, 0]

    grid = [0, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    mse = list()
    for alpha in grid:
        if alg == 'tv':
            res = TVRegDiff(y_single, itern=200, alph=alpha, u0=None, scale='small', dx=dg.solver.dt,
                            plotflag=False, precondflag=False,
                            diffkernel='sq', cgtol=1e-3, cgmaxit=100)
        elif alg == 'spline':
            res = dxdt(y_single, dg.solver.t, kind="spline", s=alpha)
        else:
            raise ValueError
        mse.append(np.sum((dxdt_hat[:, 0, 0] - res[:-1]) ** 2))

    ind = mse.index(min(mse))
    return grid[ind]


def num_diff(yt, dg, alg):
    alpha = num_dff_tuning(yt, dg, alg)

    T, B, D = yt.shape

    diff_mat = np.zeros_like(yt)
    for i in range(B):
        for j in range(D):
            y_single = yt[:, i, j]
            if alg == 'tv':
                res = TVRegDiff(y_single, itern=200, alph=alpha, u0=None, scale='small', dx=dg.solver.dt,
                                plotflag=False, precondflag=False,
                                diffkernel='sq', cgtol=1e-3, cgmaxit=100)
            elif alg == 'spline':
                res = dxdt(y_single, dg.solver.t, kind="spline", s=alpha)
            else:
                raise ValueError

            diff_mat[:, i, j] = res

    return diff_mat[:-1, :, :]


def num_diff_gp(yt, dg, ode):
    t = dg.solver.t
    noise_sigma = dg.noise_sigma
    freq = dg.freq

    X_sample_list = list()
    X_sample_list2 = list()
    pca_list = []
    # for each dimension
    assert yt.shape[-1] > 0
    for d in range(yt.shape[-1]):
        config = get_interpolation_config(ode, d)
        Y = yt[:, :, d]
        r = Y.shape[1]

        if 'sigma_in_mul' in config.keys():
            sigma_in_mul = config['sigma_in_mul']
            sigma_in = sigma_in_mul / freq
        else:
            sigma_in = config['sigma_in']

        pca = gppca.GPPCA0(r, Y, t, noise_sigma, sigma_out=ode.std_base, sigma_in=sigma_in)

        X_sample = pca.get_predictive(new_sample=1, t_new=t)
        X_sample = X_sample.reshape(len(t), X_sample.size // len(t), 1)
        X_sample_list.append(X_sample)

        X_sample2 = pca.get_predictive(new_sample=1, t_new=t+0.001)
        X_sample2 = X_sample2.reshape(len(t), X_sample2.size // len(t), 1)
        X_sample_list2.append(X_sample2)

        pca_list.append(pca)

    X_sample = np.concatenate(X_sample_list, axis=-1)
    X_sample2 = np.concatenate(X_sample_list2, axis=-1)
    if ode.positive:
        X_sample[X_sample <= 1e-6] = 1e-6
        X_sample2[X_sample2 <= 1e-6] = 1e-6
    dX = (X_sample2 - X_sample) / 0.001
    return dX[:-1, :, :], X_sample


class Interpolator:

    def __init__(self, basis, method='lstsq'):
        assert method in ['lasso', 'lstsq']
        self.basis = basis
        self.coef = None
        self.method = method

    def fit(self, X, t_obs):
        # X: T
        # t_obs: T
        design_mat = self.basis.design_matrix(t_obs)

        if self.method == 'lstsq':
            self.coef = lstsq(design_mat, X)[0]
        else:
            lasso = LassoCV(cv=5, fit_intercept=False, n_jobs=-1).fit(design_mat, X)
            self.coef = lasso.coef_

    def x_hat(self, t):
        # t: T; may be different from t_obs
        new_design_matrix = self.basis.design_matrix(t)
        return np.matmul(new_design_matrix, self.coef)

    def dxdt_hat(self, t):
        new_design_matrix = self.basis.design_matrix(t, derivative=True)
        return np.matmul(new_design_matrix, self.coef)

class InterpolatorSmoothing:
    def __init__(self):
        pass



def process_data(X, t_obs, basis, method='lstsq'):
    T, B, D = X.shape
    assert T == len(t_obs)

    interpolator_list = []
    for b in range(B):
        list_b = []
        for d in range(D):
            X_slice = X[:, b, d]
            interpolator = Interpolator(basis, method)
            interpolator.fit(X_slice, t_obs)
            list_b.append(interpolator)
        interpolator_list.append(list_b)
    return interpolator_list
