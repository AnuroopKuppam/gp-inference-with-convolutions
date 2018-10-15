from scipy.linalg import cholesky, cho_solve, cho_factor
from missing_observations import generate_data, plot
from scipy.optimize import fmin_l_bfgs_b
import numpy as np
import itertools


class naive_gp():
    def __init__(self, n, missing=False, missing_indices=[]):
        # size of the grid
        self.n = n
        # mask for missing indices
        self.missing_indices = np.array(missing_indices)
        self.missing = missing
        # variance for noise
        self.noise = 1e-6
        # variance for filling missing observations
        self.miss = 1e100
        # sqaured exponential kernel hyperparameters
        self.sigma_f = None
        self.l = None
        self.dotproducts = self._dotproducts(self.n)
        return

    # function to predict after learning the hyper parameters
    def predict(self, y):
        y = np.ndarray.flatten(y)
        K_p = self._kernel(self.n, y, self.sigma_f, self.l, mode='predict')
        K_fit = self._kernel(self.n, y, self.sigma_f, self.l, mode='fit')
        print(np.max(K_fit), np.min(K_fit))
        L, lower = cho_factor(K_fit)
        alpha = cho_solve((L, lower), y)   # K_inv.y
        return np.dot(K_p, alpha)

    # subroutine to create dot products against all indices to make the kernel matrix
    def _dotproducts(self, n):
        dot_product = np.zeros((n ** 2, n ** 2))
        x = np.arange(0, n, 1)
        y = np.arange(0, n, 1)
        iters = list(itertools.product(x, y))
        for i, list_i in enumerate(iters):
            x_i = np.array(list_i)
            for j, list_j in enumerate(iters):
                x_j = np.array(list_j)
                diff = x_i - x_j
                dot = diff.T.dot(diff)
                dot_product[i, j] = dot
        return dot_product

    # subroutine for the squared exponential kernel
    # mode: predict returns a kernel with no noise
    # mode: fit retuns a kernel with noise and missing variance
    # objective: returns the log marginal likelihood
    # grad: returns the gradients wrt to hyper parameters
    def _kernel(self, n, Y, sigma_f=0.1, l=10, mode='predict'):
        K = sigma_f ** 2 * np.exp(-1 * self.dotproducts / (2 * l ** 2))
        K_grad = np.array(K)
        if mode == 'predict':
            return K
        K += self.noise * np.eye(K.shape[0])
        if self.missing:
            indices = np.dot(np.eye(K.shape[0]), 1.0 * np.ndarray.flatten(self.missing_indices))
            K += self.miss * (np.diag(indices))
        if mode == 'fit':
            return K

        Y = np.ndarray.flatten(Y)
        L, lower = cho_factor(K)
        alpha = cho_solve((L, lower), Y)
        k_inv = cho_solve((L, lower), np.eye(L.shape[0]))

        if mode == 'objective':
            # calculate objective
            first = -0.5 * np.dot(Y.T, alpha)
            second = -1 * np.sum(np.log(np.diag(L)))
            third = -0.5 * n * np.log(2 * np.pi)
            objective = first + second + third
            return objective

        elif mode == 'grad':
            grad_sigma_f = 2 * K_grad / sigma_f
            alpha_alpha = np.dot(alpha[:, np.newaxis], alpha[:, np.newaxis].T)
            grad_sigma_f = 0.5 * np.trace(np.dot(alpha_alpha - k_inv, grad_sigma_f))
            grad_l = K_grad * (self.dotproducts / l ** 3)
            grad_l = 0.5 * np.trace(np.dot(alpha_alpha - k_inv, grad_l))
            return np.array([grad_sigma_f, grad_l])

    # calculate the objective
    def objective(self, sigma_f_l, y):
        sigma_f = sigma_f_l[0]
        l = sigma_f_l[1]
        objective = self._kernel(self.n, y, sigma_f, l, mode='objective')
        return -1.0 * objective

    # calculate the gradients for the objective
    def objective_grad(self, sigma_f_l, y):
        sigma_f = sigma_f_l[0]
        l = sigma_f_l[1]
        grad = self._kernel(self.n, y, sigma_f, l, mode='grad')
        print(sigma_f_l, grad)
        return -1.0 * grad

    # optimize wrt to the data
    def fit(self, y):
        y = np.ndarray.flatten(y)
        x, f, d = fmin_l_bfgs_b(self.objective, x0=np.array([5, 10]), fprime=self.objective_grad, args=(y,))
        self.sigma_f = x[0]
        self.l = x[1]
        print(d)
        return

    def get_params(self):
        return self.sigma_f, self.l


def main():
    temp = np.random.normal(0, 1e100, 1)
    # Y = np.array([[1, 2, 3], [2, temp, 5], [1, 2, 3]])
    # miss = np.array([[False, False, False], [False, True, False], [False, False, False]])
    Y, miss = generate_data(missing=True, miss_variance=1e100, style='plane')
    print(np.max(Y), np.min(Y))
    plot(Y)
    gp = naive_gp(Y.shape[0], missing=True, missing_indices=miss)
    gp.fit(Y)
    print(gp.get_params())
    print('XXXXXXXXXXXXX')
    predicted = gp.predict(Y)
    predicted = np.reshape(predicted, Y.shape)
    print(np.allclose(predicted, Y, atol=1e-5))
    mask = ~miss
    print(np.sum((predicted[mask] - Y[mask])**2))
    plot(predicted)
    Y[miss] = 0
    plot(Y)


if __name__ == '__main__':
    main()
