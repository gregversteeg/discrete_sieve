import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import FastICA, PCA
import sys
sys.path.append('/Users/gregv/deepcloud/discrete_sieve')
import sieve
import corex as ce


# Some utilities
from scipy.stats import rankdata

def discretize(data, k=2):
    ranks = rankdata(data, method='dense').astype(int) - 1
    j = 1
    while len(np.bincount(ranks / j)) > k:
        j += 1
    return ranks / j

def offset(data):
    data = standardize(data)
    offsets = np.cumsum(0.5 + np.max(data, axis=0))
    offsets = np.hstack([[0], offsets])[:-1]
    return data + offsets[np.newaxis, :]

def standardize(q):
    q = q.astype(float)
    delta = np.where(
        np.max(q, axis=0, keepdims=True) > np.min(q, axis=0, keepdims=True),
        np.max(q, axis=0, keepdims=True) - np.min(q, axis=0, keepdims=True),
        1)
    return (q - np.min(q, axis=0, keepdims=True)) / delta

###############################################################################
# Generate sample data
np.random.seed(0)
n_samples = 2000

s1 = np.tile([0,0,0,1,1,1,1], n_samples)[:n_samples] # Signal 1
s2 = np.tile(np.hstack([np.zeros(31, dtype=int), np.ones(6, dtype=int)]), n_samples / 10)[:n_samples] #Signal 2
s3 = np.tile([0,0,0,0,1,1,1,0,0,0,0,1,1,1,1,1,1], n_samples)[:n_samples] # Signal 3
#s1 = np.tile([0,0,1,1,2,2,2], n_samples)[:n_samples] # Signal 1
#s2 = np.tile([0,0,0,0,0,0,2,2,2,2,0,0,0,0,0,1,1,1,1,1,1,1,1], n_samples / 10)[:n_samples] #Signal 2
#s3 = np.tile([0,0,1,1,2,2,1,1,0,0,0], n_samples)[:n_samples] # Signal 3
S = np.c_[s1, s2, s3]  # Signal matrix


# Mix data
A = np.array([[1, 1, 1],
              [2, 0, -1],
              [1, 2, 0],
              [-1, 1, 0]])  # Mixing matrix
Xint = np.dot(S, A.T)  # Generate observations
Xint = Xint - np.min(Xint, axis=0, keepdims=True)  # CorEx requires non-negative ints. ICA performs whitening first anyway
# Xint[:, 3] += (np.random.random(n_samples) < 0.05).astype(int)
X = Xint.astype(float)  # ICA likes floats.

# Sieve
# out = ce.Corex(dim_hidden=3, verbose=True, n_repeat=5, smooth_marginals=False).fit(Xint)
# print zip(s1, out.labels)[:20]
# print set(zip(s1, out.labels))
# print set(zip(s2, out.labels))
# print set(zip(s3, out.labels))
# print len(set(zip(s1, out.labels)))
# print out.alpha
# print out.mis
# sys.exit()
s = sieve.Sieve(max_layers=4, dim_hidden=2, k_max=0, verbose=1, n_repeat=20, smooth_marginals=False).fit(Xint)
xbar1 = s.layers[0].transform(Xint)
xbar2 = s.layers[1].transform(xbar1)
xbar3 = s.layers[2].transform(xbar2)
# xbar4 = s.layers[3].transform(xbar3)

sieve_labels, y = s.transform(Xint)
ybar = sieve_labels[:, 3:]
xbar = sieve_labels[:, :3]
print 'MIS', s.mis


# Compute ICA
ica = FastICA(n_components=3, max_iter=10000)
S_ = ica.fit_transform(X)  # Reconstruct signals
S_ = standardize(S_)
A_ = ica.mixing_  # Get estimated mixing matrix

# For comparison, compute PCA
pca = PCA(n_components=3)
H = pca.fit_transform(X)  # Reconstruct signals based on orthogonal components
H = standardize(H)

###############################################################################
# Plot results

plt.figure(figsize=(12, 2))

models = [offset(S), offset(X), offset(H), offset(S_)] # , sieve_labels, o_labels, remainder]
names = [    'True Sources, $\\vec s$',
             'Observations, $\\vec x=A \\vec s$',
         'PCA recovered signals',
         'ICA recovered signals']
colors = ['red', 'steelblue', 'orange', 'green', 'brown', 'purple','darkgreen','grey', 'limegreen']

for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(1, 4, ii)  # Change number here
    plt.title(name, fontsize=12)
    for sig, color in zip(model.T, colors):
        plt.plot(sig[:100], '.', color=color)
        plt.ylim((-0.3, 5.3))
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)

plt.subplots_adjust(0.09, 0.04, 0.94, 0.85, 0.26, 0.46)
plt.savefig('ica.pdf')

plt.clf()

plt.figure(figsize=(10, 3))

models = map(offset, [S, X, xbar1, xbar2,xbar3 ]) #,xbar4]) # , sieve_labels, o_labels, remainder]
names =  [  'True Sources, $\\vec s$', 'Observations, $X^0$',
         'Remainder, $X^1$','Remainder, $X^2$','Remainder, $X^3$','Remainder, $X^4$']
limits = [(-0.3, 7.3), (-0.3, 7.3), (-0.3, 7.3), (-0.3, 7.3)]

for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(1, 5, ii)  # Change number here
    plt.title(name, fontsize=12)
    #if ii < 5:
    #    plt.gca().set(aspect=14)
    #else:
    #    plt.gca().set(aspect=10.)
    for sig, color in zip(model.T, colors):
        plt.plot(sig[:100], '.', color=color)
        plt.ylim((-0.3, 7.3))
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)

plt.subplots_adjust(0.09, 0.04, 0.94, 0.85, 0.26, 0.46)
plt.savefig('ica2.pdf')