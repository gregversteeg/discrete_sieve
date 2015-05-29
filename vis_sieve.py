import numpy as np
import matplotlib.pyplot as plt

# Visualization utilities
def output_dot(sieve, column_labels=None, max_edges=None, filename='structure.dot'):
    """ A network representation of the structure in Graphviz format. Units in the produced file
        are in bits. Weight is the mutual information and tc is the total correlation.
    """
    print """Compile by installing graphviz and running a command like:
             sfdp %s -Tpdf -Earrowhead=none -Nfontsize=12 \\
                -GK=2 -Gmaxiter=1000 -Goverlap=False -Gpack=True \\
                -Gpackmode=clust -Gsep=0.02 -Gratio=0.7 -Gsplines=True -o structure.pdf""" % filename
    if column_labels is None:
        column_labels = map(unicode, range(sieve.n_variables))
    else:
        column_labels = map(unicode, column_labels)
    f = open(filename, 'w')
    f.write('strict digraph {\n'.encode('utf-8'))
    for i, column_label in enumerate(column_labels):
        line = '%s [label="%s", shape=none]\n' % ('X_' + column_label, column_label)
        f.write(line.encode('utf-8'))
    for j, layer in enumerate(sieve.layers):
        this_tc = 0.6 * sieve.tcs[j] / np.max(sieve.tcs)
        line = 'Y_%d [shape=circle,margin="0,0",style=filled,fillcolor=black,' \
               'fontcolor=white,height=%0.3f,label=Y%d,tc=%0.3f]\n' % (j, this_tc, j, sieve.tcs[j] / np.log(2))
        f.write(line.encode('utf-8'))
    mis = sieve.mis
    print 'mis', mis
    print mis > 0.
    if max_edges is None or max_edges > mis.size:
        w_threshold = 0.
    else:
        w_threshold = -np.sort(-np.ravel(mis))[max_edges]
    for j, layer in enumerate(sieve.layers):
        for i in range(sieve.n_variables):
            w = mis[j, i] / np.log(2)
            if w > w_threshold:
                line = '%s -> %s [penwidth=%0.3f, weight=%0.3f];\n' % ('X_'+str(i), 'Y_'+str(j), 2 * w, w)
                f.write(line.encode('utf-8'))
        for j2 in range(0, j):
            w = mis[j, sieve.n_variables + j2] / np.log(2)
            if w > w_threshold:
                line = '%s -> %s [penwidth=%0.3f, weight=%0.3f];\n' % ('Y_'+str(j2), 'Y_'+str(j), 2 * w, w)
                f.write(line.encode('utf-8'))
    f.write('}'.encode('utf-8'))
    f.close()
    return True

def output_plots(sieve, x, column_labels=None, prefix='', topk=6):
    import os
    try:
        os.makedirs(prefix + 'relationships')
    except:
        pass
    if column_labels is None:
        column_labels = map(unicode, range(sieve.n_variables))

    for j, y in enumerate(sieve.labels.T):
        inds = np.nonzero(sieve.mis[j, :sieve.n_variables])[0]
        inds = inds[np.argsort(- sieve.mis[j, inds])[:topk]]
        if len(inds) >= 2:
            plot_rels(x[:, inds], y, map(lambda q: column_labels[q], inds),
                      prefix + 'relationships/group_num='+str(j))
    return True


def plot_rels(data, zs, labels, outfile):
    from itertools import combinations
    ns, n = data.shape
    if labels is None:
        labels = map(str, range(n))
    ncol = 5
    nrow = int(np.ceil(float(n * (n-1) / 2) / ncol))

    fig, axs = plt.subplots(nrow, ncol)
    fig.set_size_inches(5 * ncol, 5 * nrow)
    pairs = list(combinations(range(n), 2))

    for ax, pair in zip(axs.flat, pairs):
        xs, ys = data[:, pair[0]], data[:, pair[1]]
        inds = np.logical_and(xs>=0, ys>=0)
        xs, ys, zs = xs[inds], ys[inds], zs[inds]
        xset, yset = sorted(list(set(xs))), sorted(list(set(ys)))
        n = len(xs)
        ps = np.array([[np.sum((xs == x) * (ys == y)) for y in yset] for x in xset]).astype(float) / n
        hinton(ps, ax=ax)

    ax.set_xlabel(shorten(labels[pair[0]]))
    ax.set_ylabel(shorten(labels[pair[1]]))

    for ax in axs.flat[axs.size - 1:len(pairs) - 1:-1]:
        ax.scatter(data[:, 0], data[:, 1], marker='.')

    plt.rcParams['font.size'] = 12
    plt.draw()
    fig.tight_layout()
    for ax in axs.flat[axs.size - 1:len(pairs) - 1:-1]:
        ax.set_visible(False)
    fig.savefig(outfile + '.png')
    plt.close('all')
    return True


def shorten(s, n=12):
    if len(s) > 2 * n:
        return s[:n] + '..' + s[-n:]
    return s


def hinton(matrix, max_weight=None, ax=None):
    """
    Demo of a function to create Hinton diagrams.

    Hinton diagrams are useful for visualizing the values of a 2D array (e.g.
    a weight matrix): Positive and negative values are represented by white and
    black squares, respectively, and the size of each square represents the
    magnitude of each value.

    Initial idea from David Warde-Farley on the SciPy Cookbook
    """
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 1

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x,y), w in np.ndenumerate(matrix):
        color = 'black'  # TODO: add y based color here.
        size = w
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()