import numpy as np
from scipy import optimize
import sys

class Remainder(object):
    def __init__(self, xs, ys, k_max=2, deterministic=False):
        self.deterministic = deterministic
        xs, ys = np.asarray(xs), np.asarray(ys)
        self.xset, self.yset = sorted(list(set(xs))), sorted(list(set(ys)))
        k_x, k_y = len(self.xset), len(self.yset)
        self.shape = (k_max, k_x, k_y)

        counts = np.array([[np.sum((xs == x) * (ys == y)) for y in self.yset] for x in self.xset])
        self.pxy = laplace(counts)

        self.pz_xy = self.identity  # p(z | x, y), the identity for z = x
        self.identity_mi = self.mi

        if k_max == -1:
            k_max = k_x
            while True:
                self.shape = (k_max, k_x, k_y)
                self.fit()
                if self.h < 1. / len(xs) and self.mi < 0.1 * self.identity_mi:
                    break
                k_max += 1
        else:
            self.fit()

    @property
    def identity(self):
        k_max, k_x, k_y = self.shape
        pz_xy = np.zeros((k_x, k_x, k_y))
        for i in range(k_x):
            pz_xy[i, i, :] = 1.
        return pz_xy

    def marginal(self, ax, p=None, keepdims=False):
        """Marginalize over the indices in ax, z=0, x=1, y=2"""
        if p is None:
            p = self.pz_xy
        return np.sum(p * self.pxy, axis=ax, keepdims=keepdims)

    @property
    def pzy(self):
        return self.marginal(1)

    @property
    def pz(self):
        return self.marginal((1, 2))

    @property
    def mi(self):
        return self.get_mi()

    @property
    def h(self):
        return self.get_h()

    def get_mi(self, p=None):
        """ I(Z; Y), ideally this will be zero. """
        mi = entropy_f(self.marginal((1, 2), p)) + entropy_f(self.marginal((0, 1), p)) - entropy_f(self.marginal(1, p))
        return mi

    def get_h(self, p=None):
        """ H(X | Y, Z), ideally this will be zero. """
        return entropy_f(self.marginal((), p)) - entropy_f(self.marginal(1, p))

    def bias(self, p):
        """H(Z|XY), keeping this small will make the stochastic draws at the next layer more accurate."""
        return entropy_f(self.marginal((), p)) - entropy_f(self.marginal(0, p))

    def objective(self, z):
        p = z.reshape(self.shape)
        return 2 * self.get_h(p) + self.get_mi(p) + 0.01 * self.bias(p)

    def objective_jac(self, z):
        p = z.reshape(self.shape)
        pxyz = self.marginal((), p=p, keepdims=True)
        pxy = self.marginal(0, p=p, keepdims=True)
        pyz = self.marginal(1, p=p, keepdims=True)
        py = self.marginal((0, 1), p=p, keepdims=True)
        pz = self.marginal((1, 2), p=p, keepdims=True)
        return np.ravel(- pxy * (2. * (slog(pxyz) - slog(pyz))
                                 + 1. * (1 + slog(py) + slog(pz) - slog(pyz))
                                 + 0.01 * (slog(pxyz) - slog(pxy))
        )
        )

    def get_hz(self, p=None):
        """ H(X, Y, Z), ideally this will be zero. """
        return entropy_f(self.marginal((1, 2), p))

    def fit(self, n_restarts=20):
        """ Minimize 2 H(X | Y, Z) + I(Z; Y). """
        best_f = self.identity_mi  # identity transform has value I(X;Y)

        # Optimize
        mat = [
            [-1 if (j == jp and k == kp) else 0
             for _ in range(self.shape[0]) for j in range(self.shape[1]) for k in range(self.shape[2])]
            for jp in range(self.shape[1]) for kp in range(self.shape[2])
        ]
        mat = np.array(mat)
        cons = ({'type': 'eq',  # Constraints to pass into scipy optimize
                 'fun': lambda z: 1. - np.ravel(np.sum(z.reshape(self.shape), axis=0)),
                 'jac': lambda z: mat}, )
        bounds = [(0, 1) for _ in range(np.product(self.shape))]  # All probabilities between 0 and 1
        for _ in range(n_restarts):  # Multiple random initial conditions
            x0 = normalize_z(np.random.random(self.shape))
            res = optimize.minimize(self.objective, x0, jac=self.objective_jac,
                                    constraints=cons, bounds=bounds, method='SLSQP')
            if res.success and res.fun < best_f:
                best_res = res.x
                best_f = res.fun
        if best_f >= self.identity_mi:
            self.pz_xy = self.identity
            self.shape = self.identity.shape
        else:
            self.pz_xy = normalize_z(best_res.reshape(self.shape))

        self.cleanup()
        return self

    def cleanup(self):
        # Eliminate unused z values
        inds = np.sum(self.pz_xy, axis=(1, 2)) > 1e-6
        self.pz_xy = self.pz_xy[inds]

        # Sort z values in meaningful way.
        pzx = self.marginal(2)
        pz = self.marginal((1, 2))
        old_settings = np.seterr()
        np.seterr(invalid='ignore')
        px_z = pzx / pz[:, np.newaxis]
        np.seterr(**old_settings)

        exp_x = np.mean(np.array(self.xset)[np.newaxis, :] * px_z, axis=1)
        exp_x = np.where(np.isnan(exp_x), np.inf, exp_x)
        order = np.argsort(exp_x)
        self.pz_xy = self.pz_xy[order]

        # Round to deterministic solutions (will make bounds worse if k_max is not big enough)
        # I can't really see the point of this... Should probably eliminate.
        if self.deterministic:
            self.pz_xy = np.round(self.pz_xy)

    def transform(self, xs, ys):
        """ A probabilistic transform of x and y into z. We use set the seed so that we get deterministic results
        in the sense that when we transform the same data, we will always get the exact same result. (Even though
        each transformation might be probabilistic)
        """
        np.random.seed(0)  # Always use the same seed for transformation...to get deterministic results.
        return np.array([self.stochastic_label(x, y) for x, y in zip(xs, ys)])

    def predict(self, ys, zs):
        pzxy = self.marginal(())
        pzy = self.marginal(1)
        old_settings = np.seterr()
        np.seterr(invalid='ignore')
        px_yz = pzxy / pzy[:, np.newaxis, :]
        np.seterr(**old_settings)
        px = self.marginal((0, 2))
        xind_ml = np.argmax(px)

        yinds = np.array([self.yset.index(y) for y in ys])
        xinds = [np.argmax(px_yz[z, :, y])
                 if not np.any(np.isnan(px_yz[z, :, y])) else xind_ml
                 for y, z in zip(yinds, zs)]
        labels = [self.xset[xind] for xind in xinds]
        return np.array(labels)

    def stochastic_label(self, x, y):
        if x in self.xset:
            xind = self.xset.index(x)
            yind = self.yset.index(y)
            pz = self.pz_xy[:, xind, yind]
            return np.nonzero(np.random.multinomial(1, pz))[0][0]
        else:
            return -1


def normalize_z(mat):
    return mat / np.sum(mat, axis=0, keepdims=True)


def entropy_f(mat):
    """Interpret the matrix as a probabilities that sum to 1."""
    np.seterr(all='ignore')
    return np.sum(np.where(mat > 0, - mat * np.log(mat), 0))


def slog(x):
    # For entropies, 0 log 0 is always defined as 0 (and when we take derivatives?)
    # return np.log(x)
    return np.log(x.clip(1e-10))
    # return np.where(x > 0 , np.log(x), 0)


def laplace(counts, eps=0.0):
    # Calculate the probability of x and y co-occuring, with laplace smoothing.
    counts = counts.astype(float)
    n = np.sum(counts)
    # prior = np.sum(counts, axis=1, keepdims=True) * np.sum(counts, axis=0, keepdims=True) / (n * n)
    # return (counts + eps * prior) / (n + eps)
    return counts / n