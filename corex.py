"""Maximally Informative Representations using CORrelation EXplanation

Greg Ver Steeg and Aram Galstyan. "Maximally Informative
Hierarchical Representations of High-Dimensional Data"
AISTATS, 2015. arXiv preprint arXiv:1410.7404.

Code below written by:
Greg Ver Steeg (gregv@isi.edu), 2015.

License: Apache V2 (This development version not yet released)
"""

import numpy as np  # Tested with 1.8.0
from numpy import ma
from scipy.misc import logsumexp  # Tested with 0.13.0


class Corex(object):
    """
    Correlation Explanation

    A method to learn a hierarchy of successively more abstract
    representations of complex data that are maximally
    informative about the data. This method is unsupervised,
    requires no assumptions about the data-generating model,
    and scales linearly with the number of variables.

    Code follows sklearn naming/style (e.g. fit(X) to train)

    Parameters
    ----------
    n_hidden : int, optional, default=2
        Number of hidden units.

    dim_hidden : int, optional, default=2
        Each hidden unit can take dim_hidden discrete values.

    max_iter : int, optional
        Maximum number of iterations before ending.

    batch_size : int, optional
        Number of examples per minibatch. NOT IMPLEMENTED IN THIS VERSION.

    n_repeat : int, optional
        Repeat several times and take solution with highest TC.

    verbose : int, optional
        The verbosity level. The default, zero, means silent mode. 1 outputs TC(X;Y) as you go
        2 outputs MIs as you go.

    seed : integer or numpy.RandomState, optional
        A random number generator instance to define the state of the
        random permutations generator. If an integer is given, it fixes the
        seed. Defaults to the global numpy random number generator.

    Attributes
    ----------
    labels : array, [n_hidden, n_samples]
        Label for each hidden unit for each sample.

    clusters : array, [n_variables]
        Cluster label for each input variable.

    p_y_given_x : array, [n_hidden, n_samples, dim_hidden]
        The distribution of latent factors for each sample.

    mis : array, [n_hidden, n_variables]
        Mutual information between each variable and hidden unit

    tcs : array, [n_hidden]
        TC(X_Gj;Y_j) for each hidden unit

    tc : float
        Convenience variable = Sum_j tcs[j]

    tc_history : array
        Shows value of TC over the course of learning. Hopefully, it is converging.

    References
    ----------

    [1]     Greg Ver Steeg and Aram Galstyan. "Discovering Structure in
            High-Dimensional Data Through Correlation Explanation."
            NIPS, 2014. arXiv preprint arXiv:1406.1222.

    [2]     Greg Ver Steeg and Aram Galstyan. "Maximally Informative
            Hierarchical Representations of High-Dimensional Data"
            AISTATS, 2015. arXiv preprint arXiv:1410.7404.

    """
    def __init__(self, dim_hidden=2,  # Size of representations
                 batch_size=1e6, max_iter=50, n_repeat=1,  # Computational limits
                 eps=1e-5, smooth_marginals=True,    # Parameters
                 missing_values=-1, seed=None, verbose=False):

        self.dim_hidden = dim_hidden  # Each hidden factor can take dim_hidden discrete values
        self.n_hidden = 1  # Number of hidden factors to use (Y_1,...Y_m) in paper
        self.missing_values = missing_values  # For a sample value that is unknown

        self.max_iter = max_iter  # Maximum number of updates to run, regardless of convergence
        self.batch_size = batch_size  # TODO: re-implement running with mini-batches
        self.n_repeat = n_repeat  # Run multiple times and take solution with largest TC

        self.eps = eps  # Change in TC to signal convergence
        self.smooth_marginals = smooth_marginals  # Less noisy estimation of marginal distributions

        np.random.seed(seed)  # Set seed for deterministic results
        self.verbose = verbose
        if verbose > 0:
            np.set_printoptions(precision=3, suppress=True, linewidth=200)
            print 'corex, rep size:', self.n_hidden, dim_hidden
        if verbose > 1:
            np.seterr(all='warn')
        else:
            np.seterr(all='ignore')

    def label(self, p_y_given_x):
        """Maximum likelihood labels for some distribution over y's"""
        return np.argmax(p_y_given_x, axis=2)[0]

    @property
    def labels(self):
        """Maximum likelihood labels for training data. Can access with self.labels (no parens needed)"""
        return self.label(self.p_y_given_x)

    @property
    def clusters(self):
        """Return cluster labels for variables"""
        return np.argmax(self.mis[:, :, 0], axis=0)

    @property
    def tc(self):
        """The total correlation explained by all the Y's.
        """
        return np.sum(self.tcs)

    def fit(self, X):
        """Fit CorEx on the data X. See fit_transform.
        """
        self.fit_transform(X)
        return self

    def fit_transform(self, X):
        """Fit CorEx on the data

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_visible]
            The data.

        Returns
        -------
        Y: array-like, shape = [n_samples, n_hidden]
           Learned values for each latent factor for each sample.
           Y's are sorted so that Y_1 explains most correlation, etc.
        """

        Xm = ma.masked_equal(X, self.missing_values)

        best_tc = -np.inf
        for n_rep in range(self.n_repeat):

            self.initialize_parameters(X)

            for nloop in range(self.max_iter):

                self.log_p_y = self.calculate_p_y(self.p_y_given_x)
                self.theta = self.calculate_theta(Xm, self.p_y_given_x)

                log_marg_x = self.calculate_marginals_on_samples(self.theta, Xm)  # LLRs for each sample, for each var.

                self.p_y_given_x, self.log_z = self.calculate_latent(log_marg_x)

                self.update_tc(self.log_z)  # Calculate TC and record history to check convergence

                self.print_verbose()
                if self.convergence():
                    break

            if self.verbose:
                print 'Overall tc:', self.tc
            if self.tc > best_tc:
                best_tc = self.tc
                best_dict = self.__dict__.copy()
        self.__dict__ = best_dict
        if self.verbose:
            print 'Best tc:', self.tc

        self.sort_and_output(Xm)

        return self.labels

    def transform(self, X, details=False):
        """
        Label hidden factors for (possibly previously unseen) samples of data.
        Parameters: samples of data, X, shape = [n_samples, n_visible]
        Returns: , shape = [n_samples, n_hidden]
        """
        Xm = ma.masked_equal(X, self.missing_values)
        log_marg_x = self.calculate_marginals_on_samples(self.theta, Xm)
        p_y_given_x, log_z = self.calculate_latent(log_marg_x)
        labels = self.label(p_y_given_x)
        if details == 'surprise':
            # Totally experimental
            log_marg_x = self.calculate_marginals_on_samples(self.theta, Xm, return_ratio=False)
            n_samples = Xm.shape[0]
            surprise = []
            for l in range(n_samples):
                q = - sum([max([log_marg_x[j,l,i,labels[l, j]]
                                for j in range(self.n_hidden)])
                           for i in range(self.n_visible)])
                surprise.append(q)
            return p_y_given_x, log_z, np.array(surprise)
        elif details:
            return p_y_given_x, log_z
        else:
            return labels

    def initialize_parameters(self, X):
        """Set up starting state

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_visible]
            The data.

        """
        self.n_samples, self.n_visible = X.shape[:2]
        self.n_observed_xi = np.sum(X != self.missing_values, axis=0).reshape((1, -1))
        values_in_data = set(np.unique(X).tolist())-set([self.missing_values])
        self.dim_visible = int(max(values_in_data)) + 1
        if not set(range(self.dim_visible)) == values_in_data:
            print "Warning: Data matrix values should be consecutive integers starting with 0,1,..."
        self.initialize_representation()

    def calculate_p_y(self, p_y_given_x):
        """Estimate log p(y_j) using a tiny bit of Laplace smoothing to avoid infinities."""
        pseudo_counts = 0.001 + np.sum(p_y_given_x, axis=1, keepdims=True)
        log_p_y = np.log(pseudo_counts) - np.log(np.sum(pseudo_counts, axis=2, keepdims=True))
        return log_p_y

    def calculate_theta(self, Xm, p_y_given_x):
        """Estimate marginal parameters from data and expected latent labels."""
        theta = []
        for i in range(self.n_visible):
            not_missing = np.logical_not(ma.getmaskarray(Xm)[:, i])
            theta.append(self.estimate_parameters(Xm.data[not_missing, i], p_y_given_x[:, not_missing]))
        return np.array(theta)

    def calculate_latent(self, log_marg_x):
        """"Calculate the probability distribution for hidden factors for each sample."""
        log_p_y_given_x_unnorm = self.log_p_y + np.sum(log_marg_x, axis=2)
        return self.normalize_latent(log_p_y_given_x_unnorm)

    def normalize_latent(self, log_p_y_given_x_unnorm):
        """Normalize the latent variable distribution

        For each sample in the training set, we estimate a probability distribution
        over y_j, each hidden factor. Here we normalize it. (Eq. 7 in paper.)
        This normalization factor is quite useful as described in upcoming work.

        Parameters
        ----------
        Unnormalized distribution of hidden factors for each training sample.

        Returns
        -------
        p_y_given_x : 3D array, shape (n_hidden, n_samples, dim_hidden)
            p(y_j|x^l), the probability distribution over all hidden factors,
            for data samples l = 1...n_samples
        log_z : 2D array, shape (n_hidden, n_samples)
            Point-wise estimate of total correlation explained by each Y_j for each sample,
            used to estimate overall total correlation.

        """

        log_z = logsumexp(log_p_y_given_x_unnorm, axis=2)  # Essential to maintain precision.
        log_z = log_z.reshape((self.n_hidden, -1, 1))

        return np.exp(log_p_y_given_x_unnorm - log_z), log_z

    def calculate_marginals_on_samples(self, theta, Xm, return_ratio=True):
        """Calculate the value of the marginal distribution for each variable, for each hidden variable and each sample.

        theta: array parametrizing the marginals
        Xm: the data
        returns log p(y_j|x_i)/p(y_j) for each j,sample,i,y_j. [n_hidden, n_samples, n_visible, dim_hidden]
        """
        n_samples = Xm.shape[0]
        log_marg_x = np.zeros((self.n_hidden, n_samples, self.n_visible, self.dim_hidden))
        for i in range(self.n_visible):
            not_missing = np.logical_not(ma.getmaskarray(Xm)[:, i])
            log_marg_x[:, not_missing, i, :] = self.marginal_p(Xm[not_missing,i], theta[i, :, :, :])
        if return_ratio:
            # Again, I use the same p(y) here for each x_i, but for missing variables, p(y) on obs. sample may be different.
            log_p_xi = logsumexp(log_marg_x + self.log_p_y.reshape((self.n_hidden, 1, 1, self.dim_hidden)), axis=3)
            log_marg_x -= log_p_xi[:, :, :, np.newaxis]
        return log_marg_x

    def initialize_representation(self):
        self.tc_history = []
        self.tcs = np.zeros(self.n_hidden)

        p_rand = np.random.dirichlet(np.ones(self.dim_hidden), (self.n_hidden, self.n_samples))
        self.p_y_given_x, self.log_z = self.normalize_latent(np.log(p_rand))

    def update_tc(self, log_z):
        self.tcs = np.mean(log_z, axis=1).reshape(-1)
        self.tc_history.append(np.sum(self.tcs))

    def print_verbose(self):
        if self.verbose:
            print self.tcs
        if self.verbose > 1:
            print self.theta
            if hasattr(self, "mis"):
                print self.mis[:, :, 0]

    def convergence(self):
        if len(self.tc_history) > 10:
            dist = -np.mean(self.tc_history[-10:-5]) + np.mean(self.tc_history[-5:])
            return np.abs(dist) < self.eps  # Check for convergence.
        else:
            return False

    def save(self, filename):
        """ Pickle a class instance. E.g., corex.save('saved.dat') """
        import pickle
        pickle.dump(self, open(filename, 'w'), protocol=-1)

    def load(self, filename):
        """ Unpickle class instance. E.g., corex = ce.Marginal_Corex().load('saved.dat') """
        import pickle
        return pickle.load(open(filename))
    
    def sort_and_output(self, Xm):
        if not hasattr(self, 'mis'):
            # self.update_marginals(Xm, self.p_y_given_x)
            log_marg_x = self.calculate_marginals_on_samples(self.theta, Xm)  # n_hidden, n_samples, n_visible, dim_hidden
            self.mis = self.calculate_mis(self.p_y_given_x, log_marg_x)
        bias, sig = self.mi_bootstrap(Xm, n_permutation=20)
        self.mis = (self.mis - bias) * (self.mis > sig)
        self.mis = self.mis[0, :, 0]

        # Sort y values to match with the most closely correlated xi
        top_var = np.argmax(self.mis)
        pxi_y = np.exp(self.theta[top_var, :, 0, :])  # p(xi|y), dim_v by k
        exp_xi_y = np.sum(np.arange(pxi_y.shape[0])[:, np.newaxis] * pxi_y, axis=0)
        order = np.argsort(exp_xi_y)
        self.p_y_given_x = self.p_y_given_x[:, :, order]
        self.theta = self.theta[:, :, :, order]
        self.log_p_y = self.log_p_y[:, :, order]

    def calculate_mis(self, p_y_given_x, log_marg_x):
        # Could be... missing values are not properly excluded here.
        # This solution would not work with batch updates...
        mis = np.sum(p_y_given_x[:, :, np.newaxis, :] * log_marg_x, axis=(1, 3)) / self.n_observed_xi
        return mis[:, :, np.newaxis]  # MI in nats

    def mi_bootstrap(self, Xm, n_permutation=20):
        # est. if p-val < 1/n_permutation = 0.05
        mis = np.zeros((self.n_hidden, self.n_visible, n_permutation))
        for j in range(n_permutation):
            p_y_given_x = self.p_y_given_x[:, np.random.permutation(self.n_samples), :]
            theta = self.calculate_theta(Xm, p_y_given_x)
            log_marg_x = self.calculate_marginals_on_samples(theta, Xm)  # n_hidden, n_samples, n_visible, dim_hidden
            mis[:, :, j] = self.calculate_mis(p_y_given_x, log_marg_x)[:, :, 0]
        return np.mean(mis, axis=2, keepdims=True), np.sort(mis, axis=2)[:, :, [-2]]

    # MARGINAL DISTRIBUTION
    # Discrete data: should be non-negative integers starting at 0: 0,...k
    def marginal_p(self, xi, thetai):
        logp = [theta[np.newaxis, ...] for theta in thetai]  # Size dim_visible by n_hidden by dim_hidden
        return np.choose(xi.reshape((-1, 1, 1)), logp).transpose((1, 0, 2))

    def estimate_parameters(self, xi, p_y_given_x):
        x_select = (xi == np.arange(self.dim_visible)[:, np.newaxis])  # dim_v by ns
        prior = np.mean(x_select, axis=1).reshape((-1, 1, 1))  # dim_v, 1, 1
        n_obs = np.sum(p_y_given_x, axis=1)  # m, k
        counts = np.dot(x_select, p_y_given_x)  # dim_v, m, k
        p = counts + 0.001  # Tiny smoothing to avoid numerical errors
        p /= p.sum(axis=0, keepdims=True)
        if self.smooth_marginals:  # Shrinkage interpreted as hypothesis testing...
            G_stat = 2 * np.sum(np.where(counts > 0, counts * (np.log(counts) - np.log(n_obs * prior)), 0), axis=0)
            G0 = self.estimate_sig(x_select, p_y_given_x, n_obs, prior)
            z = 2
            lam = G_stat**z / (G_stat**z + G0**z)
            lam = np.where(np.isnan(lam), 0.5, lam)
            p = (1 - lam) * prior + lam * p
        return np.log(p)

    def estimate_sig(self, x_select, p_y_given_x, n_obs, prior):
        # Permute p_y_given_x, est mean Gs
        Gs = []
        for i in range(20):
            order = np.random.permutation(p_y_given_x.shape[1])
            counts = np.dot(x_select, p_y_given_x[:, order, :])  # dim_v, m, k
            Gs.append(2 * np.sum(np.where(counts > 0, counts * (np.log(counts) - np.log(n_obs * prior)), 0), axis=0))
        return np.mean(Gs, axis=0)