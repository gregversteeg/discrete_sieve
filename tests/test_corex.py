import sys
sys.path.append('..')
import corex as ce
import numpy as np
from functools import partial, update_wrapper

verbose = True
seed = 1

# Unit tests

# Random -> TC = 0

def generate_data(n_samples=100, group_sizes=[2], missing=0):
    dim_hidden = 2
    Y_true = [np.random.randint(0, dim_hidden, n_samples) for _ in group_sizes]
    X = np.hstack([np.repeat(Y_true[i][:,np.newaxis], size, axis=1) for i, size in enumerate(group_sizes)])
    clusters = [i for i in range(len(group_sizes)) for _ in range(group_sizes[i])]
    tcs = map(lambda z: (z-1) * np.log(dim_hidden), group_sizes)
    X = np.where(np.random.random(X.shape) >= missing, X, -1)
    return X, Y_true, clusters, tcs

def generate_noisy_data(n_samples=100, group_sizes=[2], erasure_p=0):
    # Implement an erasure channel with erasure probability erasure_p
    # The capacity of a single such channel is 1-erasure_p,
    # So if we have group_size < 1/(1-p) , Shannon's bound forbids perfect recovery
    # Or, 1 - 1/g <  p
    Y_true = [np.random.randint(0, 2, n_samples) for _ in group_sizes]
    X = np.hstack([np.repeat(Y_true[i][:,np.newaxis], size, axis=1) for i, size in enumerate(group_sizes)])
    rmat = np.random.random(X.shape)
    trials = np.where(rmat < erasure_p, 0, 1)
    counts = np.where(rmat < erasure_p, 0, X)  # Erasure channel
    clusters = [i for i in range(len(group_sizes)) for _ in range(group_sizes[i])]
    tcs = map(lambda z: (z-1) * np.log(2), group_sizes)
    counts = np.where(trials ==1, counts, -1)
    return counts, Y_true, clusters, tcs

def check_correct(clusters, tcs, Y_true, counts, corex):
    print corex.labels
    assert np.array_equal(corex.transform(counts), corex.labels)  # Correctness of transform
    assert np.allclose(corex.tc, np.max(tcs), atol=0.001, rtol=0.1), "TC error: %f, %f" % (corex.tc, np.max(tcs))
    true_labels = Y_true[np.argmax(tcs)]
    assert len(true_labels) == len(corex.labels)
    observed_label_correspondence = set(map(tuple, zip(corex.labels, true_labels)))
    assert len(observed_label_correspondence) == len(set(true_labels)), "Inferred labels exactly equal labels in data"
    assert set(np.where(np.array(clusters) == 0)[0]) == set(np.where(corex.mis > 0.)[0]), "MI identifies correct clusters."

def test_BinaryGaussianCorEx():
    n_samples = 100
    for group_sizes in [[2], [3, 2]]:
        np.random.seed(seed)
        counts, Y_true, clusters, tcs = generate_data(n_samples=n_samples, group_sizes=group_sizes)
        method = ce.Corex(seed=seed, verbose=verbose).fit(counts)

        f = partial(check_correct, clusters, tcs, Y_true, counts, method)
        update_wrapper(f, check_correct)
        f.description = 'groups:' + str(group_sizes) + ' seed: '+str(seed)
        yield (f, )

def test_near_shannon_limit():
    counts, Y_true, clusters, tcs = generate_noisy_data(n_samples=1000, group_sizes=[200], erasure_p=1.-3./200)
    out = ce.Corex(seed=seed, verbose=verbose).fit(counts)
    out_labels = np.rint(out.labels).astype(int)
    frac_correct = max(np.mean(Y_true[0] == out_labels), 1-np.mean(Y_true[0] == out_labels.T))
    assert frac_correct > 0.94, 'fraction correct should be high: %f' % frac_correct  # rate = 3*capacity, near perfect

    counts, Y_true, clusters, tcs = generate_noisy_data(n_samples=1000, group_sizes=[200], erasure_p=1.-1./200)
    out = ce.Corex(seed=seed, verbose=verbose).fit(counts)
    out_labels = np.rint(out.labels).astype(int)
    assert max(np.mean(Y_true[0] == out_labels), 1-np.mean(Y_true[0] == out_labels)) < 0.9  # rate=capacity, not perfect

def test_stable_solution_with_many_starting_points():
    test_data = np.repeat(np.array([[0, 0],
                          [1, 1]], dtype=int), 10, axis=0)
    n_correct = []
    for i in range(10):
        this_tc = ce.Corex(seed=i, smooth_marginals=True).fit(test_data).tc
        print this_tc
        n_correct.append(this_tc > 0.66)
    assert np.all(n_correct), "number correct %d / %d" % (np.sum(n_correct), len(n_correct))

def test_no_tc_in_random():
    sizes = [(100, 10), (200, 20)]
    tcs = []
    for size in sizes:
        test_data = np.random.randint(0, 2, size)
        tcs.append(ce.Corex(seed=seed).fit(test_data).tc)
    assert np.allclose(tcs, 0, atol=0.15), zip(sizes, tcs)

def test_mi():
    test_data = np.repeat(np.array(  [[0, 0, 0],
                                      [0, 0, 1],
                                      [0, 0, 0],
                                      [0, 0, 1],
                                      [1, 1, 0],
                                      [1, 1, 1],
                                      [1, 1, 0],
                                      [1, 1, 1]], dtype=int), 3, axis=0)
    mis = ce.Corex(seed=seed).fit(test_data).mis
    assert np.allclose(mis, np.array([np.log(2), np.log(2), 0]), atol=0.05), mis

def test_constant():
    # TODO: labels are kind of random if there is no signal... it might be nice to get that out somehow.
    for i in range(10):
        test_data = np.repeat(np.array(  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                          [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]], dtype=int), 1, axis=0)
        method = ce.Corex(seed=i).fit(test_data)
        print 'Constant data, seed %d' % i
        print 'tc, mi', method.mis, method.tc
        print 'labels', method.labels
        print 'labels', method.transform(test_data)
        assert np.array_equal(method.transform(test_data), method.labels)  # Correctness of transform
        assert np.allclose(method.tc, 0, atol=0.001, rtol=0.1), "TC error: %f, %f" % (corex.tc, np.max(tcs))
