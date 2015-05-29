import numpy as np
from scipy import optimize
import sys

class Remainder(object):
    def __init__(self, xs, ys, k_max=2, exact=True):
        xs, ys = np.asarray(xs), np.asarray(ys)
        self.xset, self.yset = sorted(list(set(xs))), sorted(list(set(ys)))
        k_x, k_y = len(self.xset), len(self.yset)
        self.shape = (k_max, k_x, k_y)
        self.n_samples = len(xs)

        counts = np.array([[np.sum((xs == x) * (ys == y)) for y in self.yset] for x in self.xset])
        self.pxy = laplace(counts)

        self.pz_xy = self.identity  # p(z | x, y), the identity for z = x
        self.identity_mi = self.mi

        PTbias = float((k_x - 1) * (k_y - 1)) / len(xs)  # Panzeri-Treves bias in MI estimate
        if self.identity_mi < PTbias:
            print 'use identity, we are within bias of zero MI'
            pass
        elif exact:
            self.pz_xy = histograms(self.pxy)
        else:
            k = 1
            old_objective = np.inf
            while True:
                self.shape = (k, k_x, k_y)
                self.fit()
                print 'k, h, mi, h(y), ident', k, self.h, self.mi, self.h_y, self.identity_mi
                if self.h < min(0.001, 1. / len(xs)) and self.mi < min(0.001, 1. / len(xs)):
                    break
                new_objective = self.objective(self.pz_xy)
                if (new_objective >= old_objective and new_objective > self.identity_mi) or k > k_max:  # No more improvement
                    print 'warning: stopping w suboptimal k because of ' \
                           'no more improvement... problems with high-d optimization.'
                    self.pz_xy = old_solution
                    self.shape = old_shape
                    break
                old_solution = self.pz_xy.copy()
                old_shape = self.shape
                old_objective = new_objective
                k += 1

        if not np.array_equal(self.pz_xy, self.identity):
            # self.cleanup(xs, ys)
            self.merge()
        assert np.allclose(np.sum(self.pz_xy, axis=0), 1), 'normalization'

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

    @property
    def h_y(self):
        return entropy_f(self.marginal((0, 1), self.identity))

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
            print 'prefer identity'
            self.pz_xy = self.identity
            self.shape = self.identity.shape
        else:
            self.pz_xy = normalize_z(best_res.reshape(self.shape))

        return self

    def merge(self):
        # Merge z's as long as I(Z;Y) is below bias.
        if self.pz_xy.shape[0] > 1:
            pz = np.sum(self.pz_xy * self.pxy, axis=(1,2))

            q = np.argmin(pz)
            p = self.pz_xy.copy()
            extra_mass = p[q]
            p = np.delete(p, q, axis=0)
            for (i, j), v in np.ndenumerate(extra_mass):
                p[np.argmax(p[:, i, j]), i, j] += v

            k_z, k_x, k_y = p.shape
            PTbias = float((k_z - 1) * (k_y - 1)) / self.n_samples  # Panzeri-Treves bias in MI estimate
            if self.get_mi(p) < PTbias and self.get_h(p) < 1e-6:
                self.pz_xy = p
                self.merge()

    def cleanup(self, xs, ys):
        # Eliminate unused z values
        inds = np.sum(self.pz_xy, axis=(1, 2)) > 1e-6
        self.pz_xy = normalize_z(self.pz_xy[inds])

        # Sort z values in meaningful way.
        # pzx = self.marginal(2)
        # pz = self.marginal((1, 2))
        # old_settings = np.seterr()
        # np.seterr(invalid='ignore')
        # px_z = pzx / pz[:, np.newaxis]
        # np.seterr(**old_settings)
        #
        # exp_x = np.mean(np.array(self.xset)[np.newaxis, :] * px_z, axis=1)
        # exp_x = np.where(np.isnan(exp_x), np.inf, exp_x)
        # order = np.argsort(exp_x)
        zs = self.transform(xs, ys)
        order = np.argsort(np.bincount(zs, minlength=self.pz_xy.shape[0]))[::-1]
        self.pz_xy = self.pz_xy[order]

    def transform(self, xs, ys):
        """ A probabilistic transform of x and y into z. We use set the seed so that we get deterministic results
        in the sense that when we transform the same data, we will always get the exact same result. (Even though
        each transformation might be probabilistic)
        """
        np.random.seed(0)  # Always use the same seed for transformation...to get deterministic results.r
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

def histograms(pxy):
    nx, ny = pxy.shape
    py = np.sum(pxy, axis=0, keepdims=True)
    px_y = (pxy / py).T
    order = np.argsort(-px_y, axis=1)  # In principle we could optimize over the order. This seemed like a good heuristic...
    px_y = np.array([px_y[i, order[i]] for i in range(ny)])
    cum = np.cumsum(px_y, axis=1)
    splits = sorted(list(set(np.ravel(cum))))  # TODO: we should merge splits if they are small (effect on h(x|zy) is within bias)
    pz_xy = np.zeros((len(splits), nx, ny))
    for i in range(nx):
        for j in range(ny):
            right = cum[j, i]
            left = 0. if i == 0 else cum[j, i-1]
            delta = right - left
            if delta > 0:
                for k in range(len(splits)):
                    z_right = splits[k]
                    z_left = 0. if k == 0 else splits[k-1]
                    z_delta = z_right - z_left
                    if z_right <= right and z_left >= left:
                        pz_xy[k, order[j][i], j] = z_delta / delta
    pz_xy = np.where(np.sum(pz_xy, axis=0) == 0, 1. / len(splits), pz_xy)
    return pz_xy


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