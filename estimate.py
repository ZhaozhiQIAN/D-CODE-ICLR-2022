import numpy as np


class ParameterEstimate:
    def __init__(self, dim_x, x_id, interpolator_lst, basis_test, ode_functional, t_int, t_weight):
        # number of equations / variables
        self.dim_x = dim_x
        # the index of variables being estimated
        self.x_id = x_id
        assert x_id < dim_x

        # interpolator_lst: list of list
        # top level list of N samples
        # second level list of P variables
        self.B = len(interpolator_lst)
        self.interpolator_lst = interpolator_lst
        for i in interpolator_lst:
            assert len(i) == dim_x

        self.basis_test = basis_test

        # take theta, generate function f(x)
        # f(x): T, B, D -> T, B
        self.ode_functional = ode_functional

        # grid and weights for integration: returned by generate integration grid
        self.t_int = t_int
        self.t_weight = t_weight
        self.t_len = len(t_int)

        # cache
        self.x_hat = np.zeros((self.t_len, self.B, self.dim_x))
        self.constants = None
        self.basis_design = None

        self.compute_x_hat_cache()
        self.compute_test_basis_cache()
        self.compute_constant_cache()

    def compute_x_hat_cache(self):
        # x_hat
        # T, B, D
        for b in range(self.B):
            for d in range(self.dim_x):
                interpolator = self.interpolator_lst[b][d]
                self.x_hat[:, b, d] = interpolator.x_hat(self.t_int)

    def compute_test_basis_cache(self):
        # generate design matrix of basis function
        # T, L
        self.basis_design = self.basis_test.design_matrix(self.t_int)

    def compute_constant_cache(self):
        # T, B
        x_hat_single = self.x_hat[:, :, self.x_id]

        # generate design matrix of basis function
        # T, L
        basis_derivative_design = self.basis_test.design_matrix(self.t_int, derivative=True)

        # B, L
        self.constants = np.matmul((x_hat_single * self.t_weight[:, None]).T, basis_derivative_design)

    def compute_loss_theta(self, theta):
        f = self.ode_functional(theta)

        # (T ; T, B, D) -> T, B
        vec_field = f(self.t_int, self.x_hat)[:, :, self.x_id]

        # B, L
        theta_pred = np.matmul((vec_field * self.t_weight[:, None]).T, self.basis_design)

        loss = np.sum((self.constants + theta_pred) ** 2)
        return loss
