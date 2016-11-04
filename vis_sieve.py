# This visualization code is a bit of a mess, cobbled together haphazardly. It just outputs graphvis graphs based on
# sieve structure.

import numpy as np


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
               'fontcolor=white,height=%0.3f,label=Y%d,tc=%0.3f]\n' % (j, this_tc, j+1, sieve.tcs[j] / np.log(2))
        f.write(line.encode('utf-8'))
    mis = sieve.mis
    print 'mis', mis
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


def shorten(s, n=12):
    if len(s) > 2 * n:
        return s[:n] + '..' + s[-n:]
    return s
