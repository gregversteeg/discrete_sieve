import sys
sys.path.append('..')
import sieve
import vis_sieve as vis
import numpy as np

verbose = True
seed = 2

def test_invertibility():
    np.random.seed(seed)
    n, ns = 3, 7
    x_count_a = np.random.randint(0, 5, (ns, 1)) + np.random.randint(0, 1, (ns, n))
    x_count_b = np.random.randint(0, 5, (ns, 1)) + np.random.randint(0, 1, (ns, n))
    x_count = np.hstack([x_count_a, x_count_b])
    out = sieve.Sieve(max_layers=2, verbose=verbose, seed=seed, k_max=7).fit(x_count)
    print 'stats for each layer'
    print [[r.h for r in layer.remainders] for layer in out.layers]
    xbar, labels = out.transform(x_count)
    x_predict = out.invert(xbar)
    print 'predicted'
    print  x_predict
    print 'actual', x_count
    assert np.allclose(x_predict, x_count)


def test_k_max():
    np.random.seed(seed)
    n, ns = 3, 300
    x_count_a = np.random.randint(0, 5, (ns, 1)) + np.random.randint(0, 1, (ns, n+1))  # NOISELESS
    x_count_b = np.random.randint(0, 5, (ns, 1)) + np.random.randint(0, 1, (ns, n))
    x_count = np.hstack([x_count_a, x_count_b])

    out = sieve.Sieve(max_layers=3, dim_hidden=5, verbose=verbose, seed=seed, k_max=8, n_repeat=10).fit(x_count)
    print len(out.layers)
    print x_count[:10]
    print out.layers[0].labels[:10]
    print out.layers[1].labels[:10]
    print 'r0 at l0', out.layers[0].remainders[0].pz_xy
    print 'r5 at l0', out.layers[0].remainders[5].pz_xy
    print ['tc: %0.3f (-) %0.3f (+) %0.3f' % (layer.corex.tc, layer.lb, layer.ub) for layer in out.layers]
    assert len(out.layers) == 2, "2 latent factors of cardinality 5 should be enough."
    xbar, labels = out.transform(x_count)
    print 'pred', out.invert(xbar)[:10]
    print 'true', x_count[:10]
    assert np.allclose(out.invert(xbar), x_count)



def test_sieve():
    test_data = np.repeat(np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                    [1, 1, 1, 1, 1, 1, 1, 1]], dtype=int), 10, axis=0)
    s = sieve.Sieve(max_layers=5, verbose=verbose, seed=seed).fit(test_data)
    assert len(s.layers) == 1, \
        'Only one layer is needed. TC and remainder at level 0: %f, %f' % (s.tc, s.lb)
    print s.transform(test_data)
    assert np.allclose(s.transform(test_data)[0][:, :-1], 0., atol=1e-4), \
        'Residual info should be small. Largest value was: %f' % np.max(np.absolute(s.transform(test_data)[0][:, :-1]))
    xbar, labels = s.transform(test_data)
    print s.invert(xbar)
    print test_data
    assert np.allclose(s.invert(xbar) - test_data, 0, atol=0.01), "Invert should be near perfect"
    assert np.allclose(s.labels, test_data[:, :1]) or np.allclose(s.labels, 1 - test_data[:, :1]), \
        'Check that labels are correct.'

#  TODO: Add a prediction test
def test_prediction():
    test_data = np.repeat(np.array(  [[-1, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [-1, 1, 1, 1, 1, 1, 1, 1],
                                      [1, 1, 1, 1, 1, 1, 1, 1],
                                      [1, 1, 1, 1, 1, 1, 1, 1]], dtype=int), 3, axis=0)
    s = sieve.Sieve(max_layers=5, verbose=verbose, seed=seed).fit(test_data)
    assert len(s.layers) == 1, \
        'Only one layer is needed. TC and remainder at level 0: %f, %f' % (s.tcs[0], s.remainders[0])  # TODO: Could fix this with more versatile Pred(xi | f(y))
    #assert np.allclose(s.transform(z)[0][0, :, :-1], 0., atol=1e-4), \
    #    'Residual info should be small. Largest value was: %f' % np.max(np.absolute(s.transform(z)[0][0, :, :-1]))
    assert np.allclose(s.labels[:, 0], test_data[:, -1]) or np.allclose(s.labels[:,0], 1 - test_data[:, -1]), \
        'Check that labels are correct.'
    xbar, labels = s.transform(test_data)
    assert np.allclose(np.where(test_data >= 0, s.invert(xbar) - test_data, 0), 0, atol=0.01), "Invert should be near perfect"
    print s.predict2(labels)[:, 0]
    print s.predict2(labels)[:, 1]
    assert np.allclose(s.predict2(labels)[:, 0], test_data[:, -1], atol=0.15), "Prediction should be close"  # Interesting reason for discrepancy here...

def test_structure():
    # Gets stuck for some seeds.
    test_data = np.repeat(np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 0, 0, 0],
                                    [1, 1, 1, 1, 1, 1, 1, 1]], dtype=int), 3, axis=0)
    s = sieve.Sieve(max_layers=5, verbose=verbose, seed=seed, n_repeat=5).fit(test_data)
    assert len(s.layers) == 2, 'Only two latent factors required.'
    assert np.all(np.argmax(s.mis, axis=0)[:-1] == np.array([0,0,0,0,0,1,1,1])), 'Correct structure has two groups.'
    assert np.allclose(s.mis[:, -1], 0), 'Latent factors should not be correlated.'

def test_vis():
    ns = 100
    xa = np.random.randint(0, 3, (ns, 1))
    xb = np.random.randint(0, 3, (ns, 1))
    xc = np.random.randint(0, 3, (ns, 1))
    test_data = np.hstack([np.repeat(xa, 7, axis=1), np.repeat(xb, 5, axis=1), np.repeat(xc, 3, axis=1)])
    s = sieve.Sieve(max_layers=4, k_max=3, dim_hidden=3, verbose=verbose, seed=seed, n_repeat=20).fit(test_data)
    vis.output_dot(s, filename='test.dot')
    vis.output_plots(s, test_data)


#  TODO: Add this test
#  r_test = lambda n, m, q: np.array([np.random.random((n,m)) < q,np.ones((n,m))] ).astype(int)
#  I've been using this to generate random data and test whether we find nonzero TC. Challenging. Of course,
#  we could always just put in a bootstrap test, but that's inelegant.