"""
Construct remainder information.

The idea is to construct z = f(x, y) so that two conditions are approximately satisfied:
I(Z:Y) = 0    (Remainder contains no info about Y)
H(X|Z, Y) = 0  (Original can be recovered)
In this code, X is the original, Y is the learned latent factor, and Z is the "remainder".
"""

import numpy as np


class Remainder(object):
    def __init__(self, xs, ys, k_max=1, strategy='predict'):
        """

        xs: a list/array of samples for x. Assumed to be discrete/categorical
        ys a list/array of samples for y
        k_max: int, default=1. The maximum increase in cardinality from x to z
        strategy: Defines the strategy for reconstructing remainder information
        """
        xs, ys = np.asarray(xs), np.asarray(ys)
        self.xset, self.yset = sorted(list(set(xs))), sorted(list(set(ys)))
        k_x, k_y = len(self.xset), len(self.yset)
        self.n_samples = len(xs)

        counts = np.array([[np.sum((xs == x) * (ys == y)) for y in self.yset] for x in self.xset])
        self.pxy = counts.astype(float) / self.n_samples

        self.pz_xy = self.identity  # p(z | x, y), the identity for z = x
        self.identity_mi = self.mi

        if strategy == 'predict' or strategy == 'lb':
            PTbias = float((k_x - 1) * (k_y - 1)) / (2 * len(xs))  # Panzeri-Treves bias in MI estimate
            if self.identity_mi < PTbias:
                print 'use identity, we are within bias of zero MI'
            else:
                self.pz_xy = exact_solution(self.pxy)  # A perfect solution but may have large k_z

            if not np.array_equal(self.pz_xy, self.identity):
                self.merge(k_x + k_max, strategy=strategy)  # Reduce k_z, at least to k_max
                self.sort_zs(xs, ys)  # Sort zs by likelihood

                if 2 * self.h + self.mi > self.identity_mi and strategy == 'predict':
                    print 'use identity, merge solution is no better'
                    self.pz_xy = self.identity
        elif strategy == 'squeeze':
            self.squeeze()
            self.squeeze()
        elif strategy == 'brute':
            order = [np.arange(k_x) for j in range(k_y)]  # identity
            p = self.permute(order)
            assert self.get_h(p=p) == 0
            print self.get_mi(p=p)


        assert np.allclose(np.sum(self.pz_xy, axis=0), 1), 'normalization'

    def merge(self, k_max, strategy='predict'):
        # Merge z's as long as I(Z;Y) is below bias.
        if self.pz_xy.shape[0] > 1:
            pz = np.sum(self.pz_xy * self.pxy, axis=(1, 2))

            if strategy == 'predict':  # The goal here is to ensure H(X|ZY) = 0, allowing small I(Y:Z)
                # Merging the following leads to a loss of accuracy in recovering xi, so we do it last.
                merge_last = np.isclose(np.max(self.pz_xy, axis=(1, 2)), 1).astype(float)
                q = np.argmin(pz + merge_last)
                p = self.pz_xy.copy()
                extra_mass = p[q]
                p = np.delete(p, q, axis=0)
                for (i, j), v in np.ndenumerate(extra_mass):
                    p[np.argmax(p[:, i, j]), i, j] += v

                k_z, k_x, k_y = p.shape
                PTbias = float((k_z - 1) * (k_y - 1)) / (2 * self.n_samples)  # Panzeri-Treves bias in MI estimate
                print 'k_z, mi, h, (bias)', k_z, self.get_mi(p), self.get_h(p), PTbias
                if (self.get_mi(p) < PTbias and self.get_h(p) < 1e-6) or k_z >= k_max:
                    self.pz_xy = p
                    self.merge(k_max, strategy=strategy)
            elif strategy == 'lb':  # The goal here is to ensure I(Y:Z)=0, allowing small H(X|YZ)
                # q = np.argmin(pz)
                #print 'hz_xy', self.hz_xy(self.pz_xy)
                q = np.argmin(self.hz_xy(self.pz_xy))
                p1 = self.pz_xy.copy()
                extra_mass = p1[q]
                p1 = np.delete(p1, q, axis=0)
                p2 = p1.copy()
                k_z, k_x, k_y = p1.shape
                p1[min(q, k_z - 1)] += extra_mass
                p2[max(0, q - 1)] += extra_mass
                p = p1 if self.get_h(p1) < self.get_h(p2) else p2

                PTbias = float((k_z - 1) * (k_y - 1)) / (2 * self.n_samples)  # Panzeri-Treves bias in MI estimate
                #print 'k_z, mi, h, (bias)', k_z, self.get_mi(p), self.get_h(p), PTbias
                if k_z >= k_max:
                    self.pz_xy = p
                    self.merge(k_max, strategy=strategy)

    def squeeze(self):
        # Try re-ordering x's to reduce MI
        pxy = self.pxy
        k_x, k_y = pxy.shape
        py = np.sum(pxy, axis=0)  # p(y)
        squeeze_order = np.argsort(-py)
        p = self.pz_xy.copy()
        for j in squeeze_order[1:]:
            best_roll = []
            for dir in [1, -1]:
                for l in range(k_x):
                    this_order = np.roll(np.arange(k_x)[::dir], l)
                    p[:, :, j] = self.pz_xy[:, this_order, j]
                    v = self.get_mi(p)
                    best_roll.append((v, this_order))
            v, this_order = min(best_roll, key=lambda z: z[0])
            #print 'best_roll', best_roll
            #print 'v, order', v, this_order
            self.pz_xy[:, :, j] = self.pz_xy[:, this_order, j]
            #print 'improvement?', self.identity_mi > self.mi, self.identity_mi, self.mi, self.h, this_order

    def sort_zs(self, xs, ys):
        zs = self.transform(xs, ys)
        order = np.argsort(np.bincount(zs, minlength=self.pz_xy.shape[0]))[::-1]
        self.pz_xy = self.pz_xy[order]

    def hz_xy(self, p):
        return np.sum(- p * self.pxy * np.where(p > 0, np.log(p), 0), axis=(1,2))

    @property
    def identity(self):
        k_x, k_y = self.pxy.shape
        pz_xy = np.zeros((k_x, k_x, k_y))
        for i in range(k_x):
            pz_xy[i, i, :] = 1.
        return pz_xy

    def permute(self, order):
        k_x, k_y = self.pxy.shape
        pz_xy = np.zeros((k_x, k_x, k_y))
        for j in range(k_y):
            for i in range(k_x):
                pz_xy[i, order[j][i], j] = 1.
        return pz_xy

    def marginal(self, ax, p=None, keepdims=False):
        """Marginalize over the indices in ax, z=0, x=1, y=2"""
        if p is None:
            p = self.pz_xy
        return np.sum(p * self.pxy, axis=ax, keepdims=keepdims)

    @property
    def mi(self):
        return self.get_mi()

    def get_mi(self, p=None):
        """ I(Z; Y), ideally this will be zero. """
        mi = entropy_f(self.marginal((1, 2), p)) + entropy_f(self.marginal((0, 1), p)) - entropy_f(self.marginal(1, p))
        return mi

    @property
    def h(self):
        return self.get_h()

    def get_h(self, p=None):
        """ H(X | Y, Z), ideally this will be zero. """
        return entropy_f(self.marginal((), p)) - entropy_f(self.marginal(1, p))

    def transform(self, xs, ys):
        """ A probabilistic transform of x and y into z. We use set the seed so that we get deterministic results
        in the sense that when we transform the same data, we will always get the exact same result. (Even though
        each transformation might be probabilistic)
        """
        np.random.seed(0)  # Always use the same seed for transformation...to get deterministic results.r
        return np.array([self.stochastic_label(x, y) for x, y in zip(xs, ys)])

    def predict(self, ys, zs):
        return np.array([self.predict_one(y, z) for y, z in zip(ys, zs)])

    def predict_one(self, y, z):
        if not hasattr(self, 'p_dict'):  # Memoize the predictions.
            self.p_dict = {}
        if (y, z) in self.p_dict:
            return self.p_dict[(y, z)]
        pzxy = self.marginal(())
        pzy = self.marginal(1)
        old_settings = np.seterr()
        np.seterr(invalid='ignore')
        px_yz = pzxy / pzy[:, np.newaxis, :]
        np.seterr(**old_settings)
        px = self.marginal((0, 2))
        xind_ml = np.argmax(px)

        yind = self.yset.index(y)
        xind = np.argmax(px_yz[z, :, yind]) if not np.any(np.isnan(px_yz[z, :, yind])) else xind_ml
        label = self.xset[xind]
        self.p_dict[(y, z)] = label
        return label

    def stochastic_label(self, x, y):
        if x in self.xset:
            xind = self.xset.index(x)
            yind = self.yset.index(y)
            pz = self.pz_xy[:, xind, yind]
            return np.nonzero(np.random.multinomial(1, pz))[0][0]
        else:
            return -1


def zero_mi(pxy):
    py = np.sum(pxy, axis=0, keepdims=True)  # p(y)
    px_y = (pxy / py).T  # p(x|y)
    othery = range(k_y)
    topy = othery.pop(np.argmax(py))
    splits = np.cumsum(px_y[topy])
    pz_xy = np.zeros((k_x, k_x, k_y))  # Now we fill in p(z|xy)
    pz_xy[:, :, topy] = np.diag(np.ones(k_x))
    for j in othery:

        pz_xy[:, :, j] = 0  # TODO: Give up on this one?
        # Now for each y, we'll try to line it up so that H(x|y=y, z) is minimized

    pz_xy = np.where(np.sum(pz_xy, axis=0) == 0, 1. / len(splits), pz_xy)
    return pz_xy

def exact_solution(pxy):
    """This is my default way of writing code: procedural, dense, and impenetrable. Anyway, the idea here is that
        we look for p(z|xy) so that I(z:y) = 0 and H(x|yz) = 0. This requires us to make the cardinality of z
        potentially quite big. In a separate function we try to
        reduce the cardinality of z while maintaining our desired properties.
     """
    py = np.sum(pxy, axis=0, keepdims=True)  # p(y)
    px_y = (pxy / py).T  # p(x|y)
    order = order1(px_y, py)  # In principle we could optimize over the order. This is a good heuristic...
    pz_xy = solution_from_order(px_y, order)
    return pz_xy


def solution_from_order(px_y, order):
    k_y, k_x = px_y.shape
    px_y = np.array([px_y[i, order[i]] for i in range(k_y)])  # sort largest probability mass of x, for each y
    cum = np.cumsum(px_y, axis=1)  # Cumulative distribution (of the sorted probabilities of x for each y)
    splits = sorted(list(set(np.ravel(cum))))  # We define a split in z at each histogram boundary of x|y
    pz_xy = np.zeros((len(splits), k_x, k_y))  # Now we fill in p(z|xy)
    for i in range(k_x):
        for j in range(k_y):
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


def order0(px_y, py):
    k_y, k_x = px_y.shape
    order = np.array([np.arange(k_x) for _ in range(k_y)])
    return order


def order1(px_y, py):
    return np.argsort(-px_y, axis=1)


def order2(px_y, py):
    k_y, k_x = px_y.shape
    othery = range(k_y)
    topy = othery.pop(np.argmax(py))
    order = np.array([np.arange(k_x) for _ in range(k_y)])
    for j in othery:
        # Try to line up with first one
        best_roll = []
        for dir in [1, -1]:
            for l in range(k_x):
                this_order = np.roll(order[j][::dir], l)
                v = np.sum(np.abs(np.cumsum(px_y[j, this_order]) - np.cumsum(px_y[topy])))
                print 'test!', this_order, v, dir
                best_roll.append((v, l, dir))
        best_v, best_l, best_dir = min(best_roll)
        order[j] = np.roll(order[j][::dir], best_l)
    return order

def order3(pxy):
    # Try re-ordering x's to reduce MI
    k_x, k_y = pxy.shape
    py = np.sum(pxy, axis=0)  # p(y)
    squeeze_order = np.argsort(-py)
    p0 = identity(k_x, k_y)
    p = p0.copy()
    orders = [np.arange(k_x)]
    for j in squeeze_order[1:]:
        best_roll = []
        for dir in [1, -1]:
            for l in range(k_x):
                this_order = np.roll(np.arange(k_x)[::dir], l)
                p[:, :, j] = p0[:, this_order, j]
                v = get_mi(p, pxy)
                best_roll.append((v, this_order))
        v, this_order = min(best_roll, key=lambda z: z[0])
        #print 'best_roll', best_roll
        #print 'v, order', v, this_order
        orders.append(this_order)
        p[:, :, j] = p0[:, this_order, j]
        print 'improvement?', [v for v, o in best_roll], get_mi(p, pxy), this_order
    return np.array(orders)

def entropy_f(mat):
    """Interpret the matrix as a probabilities that sum to 1."""
    np.seterr(all='ignore')
    return np.sum(np.where(mat > 0, - mat * np.log(mat), 0))

def identity(k_x, k_y):
    pz_xy = np.zeros((k_x, k_x, k_y))
    for i in range(k_x):
        pz_xy[i, i, :] = 1.
    return pz_xy

def marginal(ax, pz_xy, pxy, keepdims=False):
    """Marginalize over the indices in ax, z=0, x=1, y=2"""
    return np.sum(pz_xy * pxy, axis=ax, keepdims=keepdims)

def get_mi(pz_xy, pxy):
    """ I(Z; Y), ideally this will be zero. """
    mi = entropy_f(marginal((1, 2), pz_xy, pxy)) + entropy_f(marginal((0, 1), pz_xy, pxy)) \
         - entropy_f(marginal(1, pz_xy, pxy))
    return mi

def get_h(pz_xy, pxy):
    """ H(X | Y, Z), ideally this will be zero. """
    return entropy_f(marginal((), pz_xy, pxy)) - entropy_f(marginal(1, pz_xy, pxy))

