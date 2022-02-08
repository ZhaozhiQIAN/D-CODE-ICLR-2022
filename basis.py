import numpy as np
import abc
from scipy.interpolate import BSpline

class BasisFunction(metaclass=abc.ABCMeta):
    def __init__(self, T):
        # scaling with max time
        self.T = T
        self.current_basis = None
        self.n_basis = None

    def set_current_basis(self, i):
        assert i < self.n_basis
        self.current_basis = i

    @abc.abstractmethod
    def phi_t(self, t):
        """
        Evaluate basis function with order at time t
        """
        pass

    @abc.abstractmethod
    def dphi_dt(self, t):
        """
            Evaluate the derivative of the basis function with order at time t
        """
        pass

    @abc.abstractmethod
    def design_matrix(self, t, derivative=False):
        """
            Evaluate the derivative of the basis function with order at time t
        """
        pass

    @abc.abstractmethod
    def get_nonzero_range(self):
        """
            get non zero range [lower, upper] for a given basis function in the family
        """
        pass


class FourierBasis(BasisFunction):
    def __init__(self, T, order):
        super().__init__(T)
        self.sqrt_2 = np.sqrt(2)
        self.n_basis = order
        self.name = 'FourierBasis'

    def set_current_basis(self, i):
        # [0, n_basis)
        assert i < self.n_basis
        self.current_basis = i + 1

    def phi_t(self, t):
        t = t / self.T
        return self.sqrt_2 * np.sin(self.current_basis * np.pi * t)

    def dphi_dt(self, t):
        t = t / self.T
        return self.sqrt_2 * np.cos(self.current_basis * np.pi * t) * self.current_basis * np.pi / self.T

    def design_matrix(self, t, derivative=False):
        # t: time steps of observations

        cols = []
        save = self.current_basis

        for i in range(self.n_basis):
            self.current_basis = i + 1
            if not derivative:
                cols.append(self.phi_t(t))
            else:
                cols.append(self.dphi_dt(t))

        mat = np.stack(cols, axis=-1)
        self.current_basis = save
        return mat

    def get_nonzero_range(self):
        return 0, self.T


class CubicSplineBasis(BasisFunction):
    def __init__(self, T, freq, zero_constraint=True):
        super().__init__(T)
        self.name = 'CubicSplineBasis'
        degree = 3
        freq += 2
        self.zero_constraint = zero_constraint
        self.dt = T / freq
        if zero_constraint:
            self.grid = np.arange(0, T + self.dt, self.dt)
        else:
            self.grid = np.arange(0 - degree * self.dt, T + (1 + degree) * self.dt, self.dt)
        self.basis = []
        self.basis_derivative = []
        for i in range(0, len(self.grid) - degree - 1):
            knots = self.grid[i:(i + degree + 2)]
            b = BSpline.basis_element(knots, extrapolate=False)
            self.basis.append(b)
            self.basis_derivative.append(b.derivative())

        self.n_basis = len(self.basis)
        self.norm = np.zeros(self.n_basis)
        self.norm_flag = False

    def phi_t(self, t):
        return self.basis[self.current_basis](t)

    def dphi_dt(self, t):
        return self.basis_derivative[self.current_basis](t)

    def design_matrix(self, t, derivative=False):
        # t: time steps of observations
        mat = np.zeros((len(t), self.n_basis))

        save = self.current_basis

        for j in range(self.n_basis):
            self.set_current_basis(j)
            s, e = self.get_nonzero_range()
            inds = np.nonzero((t >= s) & (t <= e))[0]
            if not derivative:
                mat[inds, j] = self.phi_t(t[inds])
            else:
                mat[inds, j] = self.dphi_dt(t[inds])
        self.current_basis = save

        assert mat.shape == (len(t), self.n_basis)
        if not derivative:
            self.norm = np.sqrt(np.mean(mat ** 2, axis=0))
            self.norm_flag = True

        assert self.norm_flag
        mat = mat / self.norm[None, :]
        return mat

    def get_nonzero_range(self):
        basis = self.basis[self.current_basis]
        return basis.t[basis.k], basis.t[-basis.k-1]
