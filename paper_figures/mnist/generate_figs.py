import numpy as np
import pylab
import sys, pickle, cPickle, gzip

sys.path.append('/Users/gregv/deepcloud/discrete_sieve')
import sieve
import vis_sieve

def entropy(x):
  n_samples = len(x)
  ps = np.bincount(x).astype(float) / n_samples
  return np.sum(np.where(ps > 0, -ps * np.log2(ps), 0))

def save_digit(z, filename, cmap=pylab.cm.gray):
    pylab.clf()
    pylab.axis('off')
    pylab.imshow(z.reshape((28, 28)), interpolation='nearest', cmap=cmap, vmin=-1, vmax=1)
    pylab.savefig('results/' + filename + '.pdf')
    pylab.clf()

def stack_digit(zs, filename):
    pylab.clf()
    fig = pylab.figure(frameon=False)
    fig.set_size_inches(1, 3)
    ax = pylab.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(np.vstack(zs).reshape((-1, 28)), interpolation='nearest', cmap=pylab.cm.gray)
    fig.savefig('results/' + filename + '.pdf')
    pylab.close('all')

def vis_components(s, xbar, labels):
    n_layers = len(s.layers)
    # labels = labels[:10]
    n_samples = len(labels)
    components = np.zeros((n_layers, 784))
    # components = []
    # for l in range(n_layers):
    #     print 'l', l
    #     labels0, labels1 = labels.copy(), labels.copy()
    #     labels0[:,l] = 0
    #     labels1[:,l] = 1
    #     effect = [np.mean((s.predict_variable(labels1, i) - s.predict_variable(labels0, i)).astype(float))
    #      for i in range(784)]
    #     components.append(np.array(effect))


    for i in range(784):
        print 'i', i
        for zi, y in zip(xbar[:, i], labels):
            for l in range(n_layers):
                xi = s.invert_variable(zi, y, i)
                y_prime = y.copy()
                y_prime[l] = 1 - y_prime[l]
                xi_prime = s.invert_variable(zi, y_prime, i)
                components[l,i] += ((xi_prime - xi) * (-1)**y[l]).astype(float) / n_samples

    cPickle.dump(components, open('components.dat', 'w'), protocol=-1)


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()
train_data = (train_set[0] >= 0.5).astype(int)
test_data = (test_set[0] >= 0.5).astype(int)
train_labels = train_set[1]
test_labels = train_labels[1]

n_test = 100

a = pickle.load(open('sieve_k_max=0.dat'))
print 'transforming'
try:
    xbar, labels = cPickle.load(open('xbar,labels.dat'))
except:
    xbar, labels = a.transform(train_data)
    cPickle.dump((xbar, labels), open('xbar,labels.dat', 'w'), protocol=-1)
    pickle.dump(a, open('sieve_k_max=1_dict.dat', 'w'))

print 'compression'
this_x = train_data
for l, layer in enumerate(a.layers):
  entropies = [entropy(x) for x in this_x.T]
  print '%d, %0.3f' %(l, sum(entropies))
  this_x = layer.transform(this_x)

sys.exit()

print 'visualize components'
vis_components(a, xbar, labels)
components = cPickle.load(open('components.dat'))
for l, c in enumerate(components):
    save_digit(c, 'components/%d'%l, cmap=pylab.cm.seismic)

sys.exit()

np.random.seed(1)
random_sub = np.random.choice(np.arange(train_data.shape[0]), n_test)

print 'layers', len(a.layers)
print 'tc: %0.3f, ub(+): %0.3f, lb(-): %0.3f' % (a.tc, a.ub, a.lb)

vis_sieve.output_dot(a, max_edges=300, filename='mnist.dot')
#print 'Originals'
#for l, z in zip(random_sub, train_data[random_sub]):
#    save_digit(z, '/originals/%d' % l)

print 'Hallucinate'
n_test = 500
xbar_hal = np.vstack([np.random.choice(xbar[:, i], n_test) for i in range(xbar.shape[1])]).T

#xbar_hal, labels_hal = a.transform(np.random.randint(0, 2, (n_test, 784)))
data_hal = np.zeros((n_test, 784), dtype=float)
for l in range(n_test):
    xbar_hal = np.hstack([np.roll(xbar_hal[:,:784], 1, axis=0), xbar_hal[:,784:]])
    data_hal += a.invert(xbar_hal).astype(float) / n_test

for l in range(n_test):
    z = data_hal[l]
    #neighbor_index = np.argmax(np.sum((z>0.5) == train_data.astype(bool), axis=1))
    #nnz = train_data[neighbor_index]
    save_digit((z>=0.5).astype(int), '/hallucinate/%d' % l)

print 'Lossy/orig/inpaint'
labels_sub = labels[random_sub]
lossy = a.predict(labels_sub)
sub_train = train_data[random_sub]
sub_train[:, 392:] = -1
xbar_missing, labels_missing = a.transform(sub_train)
sub_paint = a.predict(labels_missing)
knit = np.hstack([sub_train[:, :392], sub_paint[:,392:]]).astype(float)
knit[:,392:] = np.clip(knit[:,392:], 0.4, 1)

for il, l in enumerate(random_sub):
    stack_digit([train_data[l], lossy[il], knit[il]], '/lossy/%d' % l)


