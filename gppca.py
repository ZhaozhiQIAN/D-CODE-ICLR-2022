import numpy as np
from scipy.optimize import line_search, bracket
from scipy.optimize import minimize


def get_rbf_kernel(t, sigma_out, sigma_in, t2=None):
    tc = t[:, None]
    if t2 is None:
        tr = t
    else:
        tr = t2
    t_mat = sigma_out ** 2 * np.exp(-1. / (2 * sigma_in ** 2) * (tc - tr) ** 2)
    return t_mat


class GPPCA0:
    def __init__(self, r, Y, t, sigma, sigma_out=None, sigma_in=None):
        self.r = r
        self.Y = Y
        self.t = t
        self.sigma = sigma
        self.n_traj = Y.shape[1]

        if sigma_out is None:
            self.sigma_out = np.std(Y)
        else:
            self.sigma_out = sigma_out

        if sigma_in is None:
            self.sigma_in = self.t[1] - self.t[0]
        else:
            self.sigma_in = sigma_in

        self.K = self.get_kernel_discrete()
        self.A = self.get_factor_loading()

    def get_hyper_param(self, method='Powell'):
        x0 = np.log(np.array([self.sigma_in]))
        res = minimize(self.loss_fn, x0=x0, method=method)
        # print(res)
        self.sigma_in = np.exp(res['x'][0])

    def loss_fn(self, x):
        # input in log scale
        sigma_out = self.sigma_out
        sigma_in = x[0]
        sigma_in = np.exp(sigma_in)

        tau = sigma_out ** 2 / self.sigma ** 2

        K = get_rbf_kernel(self.t, sigma_out, sigma_in)

        # T, T
        W = np.linalg.inv(1. / tau * np.linalg.inv(K) + np.eye(K.shape[0]))
        # R, T
        b = np.matmul(self.Y, self.A).T

        S = np.abs(np.sum(self.Y ** 2) - np.sum(np.diag(b @ W @ b.T)))

        f2 = np.log(S) * (-1 * self.Y.shape[0] * self.Y.shape[1] / 2)

        f1 = -1. / 2 * self.r * np.linalg.slogdet(tau * K + np.eye(K.shape[0]))[1]

        return -1 * (f1 + f2)

    def get_predictive(self, new_sample=1, t_new=None):
        if new_sample == 1:
            return self.get_predictive_mean(1, t_new)
        # predictive distribution
        Z_hat = self.get_factor(t_new=t_new)
        X_hat = self.get_X_mean(Z_hat)
        K_fac = self.get_X_cov(t_new=t_new)

        X_list = list()
        for i in range(new_sample):
            noise = self.sample_noise(K_fac)
            X_sample = X_hat + noise
            X_sample = X_sample[:, None, :]
            X_list.append(X_sample)

        X = np.concatenate(X_list, axis=1)
        return X

    def get_predictive_mean(self, new_sample=1, t_new=None):
        assert new_sample == 1
        # predictive distribution
        Z_hat = self.get_factor(t_new=t_new)
        X_hat = self.get_X_mean(Z_hat)

        return X_hat

    def get_factor_loading(self):
        G = self.get_G()
        w, v = np.linalg.eigh(G)
        A = v[:, -self.r:]
        return A

    def get_X_cov(self, t_new=None):
        if t_new is None:
            K = self.K
        else:
            K = get_rbf_kernel(t_new, self.sigma_out, self.sigma_in, t_new)

        # T, D
        D = self.sigma ** 2 * np.linalg.inv(1. / self.sigma ** 2 * np.linalg.inv(K) + np.eye(K.shape[0]))
        try:
            D_fac = np.linalg.cholesky(D)
        except np.linalg.LinAlgError:
            w, v = np.linalg.eigh(D)
            w_pos = w
            w_pos[w_pos < 0] = 0
            w_pos = np.sqrt(w_pos)
            D_fac = v @ np.diag(w_pos)

        # B, R
        A_fac = self.A
        K_fac = np.kron(D_fac, A_fac)
        return K_fac

    def sample_noise(self, K_fac):
        noise = np.random.randn(K_fac.shape[1])
        vec = K_fac @ noise
        mat = vec.reshape(len(vec) // self.n_traj, self.n_traj)
        return mat

    def get_factor(self, t_new=None):
        if t_new is None:
            f1 = self.K
        else:
            f1 = get_rbf_kernel(t_new, self.sigma_out, self.sigma_in, self.t)

        # print(f1.shape)
        # print(self.K.shape)
        Z_hat = f1 @ np.linalg.inv(self.K + self.sigma ** 2 * np.eye(self.K.shape[0])) @ self.Y @ self.A
        return Z_hat

    def get_X_mean(self, Z_hat):
        X_hat = np.matmul(Z_hat, self.A.T)
        return X_hat

    def get_kernel_discrete(self):
        sigma_out = self.sigma_out
        sigma_in = self.sigma_in
        K = get_rbf_kernel(self.t, sigma_out, sigma_in)
        return K

    def get_G(self):
        # get G matrix - Thm 3 Eq 7

        W = np.linalg.inv(self.sigma ** 2 * np.linalg.inv(self.K) + np.eye(self.K.shape[0]))
        G = np.matmul(np.matmul(self.Y.transpose(), W), self.Y)
        return G



# class GPPCA:
#     def __init__(self, r, Y, t, sigma, sigma_out_arr=None, sigma_in_arr=None, atol=1E-3, a_max_itr=1000):
#         self.r = r
#         self.Y = Y
#         self.t = t
#         self.sigma = sigma
#         self.atol = atol
#         self.a_max_itr = a_max_itr
#
#         if sigma_out_arr is None:
#             self.sigma_out_arr = np.random.uniform(0.1, 0.6, r)
#         else:
#             self.sigma_out_arr = sigma_out_arr
#
#         if sigma_in_arr is None:
#             self.sigma_in_arr = np.random.uniform(0.5, 1.5, r)
#         else:
#             self.sigma_in_arr = sigma_in_arr
#
#         # initialize X
#         self.X = self.get_init_X()
#         # set up problem
#         self.G_list = self.get_G_list()
#
#     # def get_Z_hat(self):
#     #     K = get_rbf_kernel(self.t, sigma_out, sigma_in)
#     #     Z_hat = np.matmul(np.matmul(np.matmul(K, np.linalg.inv(K + self.sigma ** 2 * np.eye(K.shape[0]))), self.Y), self.X)
#     #
#
#     def optimize_factor_loading(self):
#         for i in range(self.a_max_itr):
#             # print(i)
#             ret = self.update_X(i)
#             if ret == 0:
#                 break
#
#     def update_X(self, itr):
#         # Get A matrix - Eq 4
#
#         Grad_list = []
#
#         for i in range(self.r):
#             Grad_list.append(-2. * self.G_list[i] @ self.X[:, i])
#
#         Grad = np.stack(Grad_list, axis=1)
#
#         total_grad = np.sum(Grad ** 2)
#         # print(total_grad)
#         if total_grad < self.atol:
#             return 0
#
#         A = Grad @ self.X.T - self.X @ Grad.T
#
#         # line search
#         X = self.X
#         r = self.r
#         G_list = self.G_list
#
#         def get_x_new(tau):
#             Q = np.linalg.inv(np.eye(A.shape[0]) + tau / 2 * A) @ (np.eye(A.shape[0]) - tau / 2 * A)
#             return Q @ X
#
#         def line_search_fn(tau):
#             Q = np.linalg.inv(np.eye(A.shape[0]) + tau / 2 * A) @ (np.eye(A.shape[0]) - tau / 2 * A)
#             X_new = Q @ X
#             # evaluate objective function
#             obj = 0
#
#             for i in range(r):
#                 x = X_new[:, i][:, None]
#                 obj += -1. * x.T @ G_list[i] @ x
#
#             return obj
#
#         delta_h = 1E-5
#
#         def line_search_fn_gard(tau):
#
#             y0 = line_search_fn(tau - delta_h)
#             y1 = line_search_fn(tau + delta_h)
#             return (y1 - y0) / delta_h / 2
#
#         # xk = np.array([0.])
#         # pk = np.array([1.0])
#         #
#         # ls_res = line_search(line_search_fn, line_search_fn_gard, xk=xk, pk=pk)
#         # print(ls_res)
#         # tau_star = ls_res[0]
#
#         xa, xb, xc, fa, fb, fc, funcalls = bracket(line_search_fn, xa=0., xb=1.)
#         tau_star = xb
#
#         if tau_star is None or tau_star == 0:
#             return 0
#
#         self.X = get_x_new(tau_star)
#         if itr % 100 == 0:
#             print(itr)
#             obj = 0
#
#             for i in range(r):
#                 x = self.X[:, i][:, None]
#                 obj += -1. * x.T @ G_list[i] @ x
#             print(obj[0, 0])
#
#         return 1
#
#     def get_init_X(self):
#
#         n, m = self.Y.shape[1], self.r
#
#         H = np.random.randn(n, m)
#         u, s, vh = np.linalg.svd(H, full_matrices=False)
#         # T, r
#         return u @ vh
#
#     def get_G_list(self):
#         # get G matrix - Thm 3 Eq 7
#
#         G_list = []
#
#         for sigma_out, sigma_in in zip(self.sigma_out_arr, self.sigma_in_arr):
#             K = get_rbf_kernel(self.t, sigma_out, sigma_in)
#             # remove the following debug code
#             # K = np.eye(K.shape[0]) * 100
#
#             W = np.linalg.inv(self.sigma ** 2 * np.linalg.inv(K) + np.eye(K.shape[0]))
#             G = np.matmul(np.matmul(self.Y.transpose(), W), self.Y)
#             G_list.append(G)
#         return G_list
#
