import numpy as np


class Remainder(object):
    def __init__(self, xs, ys, k_max=np.inf):
        xs, ys = np.asarray(xs), np.asarray(ys)
        self.xset, self.yset = sorted(list(set(xs))), sorted(list(set(ys)))
        k_x, k_y = len(self.xset), len(self.yset)
        self.n_samples = len(xs)

        counts = np.array([[np.sum((xs == x) * (ys == y)) for y in self.yset] for x in self.xset])
        self.pxy = counts.astype(float) / self.n_samples

        self.pz_xy = self.identity  # p(z | x, y), the identity for z = x
        self.identity_mi = self.mi

        PTbias = float((k_x - 1) * (k_y - 1)) / (2 * len(xs))  # Panzeri-Treves bias in MI estimate
        if self.identity_mi < PTbias:
            print 'use identity, we are within bias of zero MI'
        else:
            self.pz_xy = histograms(self.pxy)

        if not np.array_equal(self.pz_xy, self.identity):
            self.merge(k_max=k_max)
            self.sort_zs(xs, ys)
        assert np.allclose(np.sum(self.pz_xy, axis=0), 1), 'normalization'

    def merge(self, k_max=np.inf):
        # Merge z's as long as I(Z;Y) is below bias.
        if self.pz_xy.shape[0] > 1:
            pz = np.sum(self.pz_xy * self.pxy, axis=(1,2))

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
            print 'test', k_z, self.get_mi(p), PTbias, self.get_h(p)
            if (self.get_mi(p) < PTbias and self.get_h(p) < 1e-6) or k_z > k_max:
                self.pz_xy = p
                self.merge(k_max=k_max)

    def sort_zs(self, xs, ys):
        zs = self.transform(xs, ys)
        order = np.argsort(np.bincount(zs, minlength=self.pz_xy.shape[0]))[::-1]
        self.pz_xy = self.pz_xy[order]

    @property
    def identity(self):
        k_x, k_y = self.pxy.shape
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
    """This is my default way of writing code: procedural, dense, and impenetrable. Anyway, the idea here is that
        we look for p(z|xy) so that I(z:y) = 0 and H(x|yz) = 0. This requires us to make the cardinality of z
        potentially quite big. In a separate function we try to
        reduce the cardinality of z while maintaining our desired properties.
     """
    k_x, k_y = pxy.shape
    py = np.sum(pxy, axis=0, keepdims=True)  # p(y)
    px_y = (pxy / py).T  # p(x|y)
    order = np.argsort(-px_y, axis=1)  # In principle we could optimize over the order. This is a good heuristic...
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


def entropy_f(mat):
    """Interpret the matrix as a probabilities that sum to 1."""
    np.seterr(all='ignore')
    return np.sum(np.where(mat > 0, - mat * np.log(mat), 0))