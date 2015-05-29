import sys
sys.path.append('..')
import remainder as re
import numpy as np

def test_null():
    xs = [0, 0, 0, 0, 0, 0]
    ys = [0, 0, 0, 0, 0, 0]
    g = re.Remainder(xs, ys, k_max=2)
    assert np.isclose(g.mi, 0), "No MI possible: %0.7f > 0" % g.mi
    assert np.isclose(g.h,0), "No uncertainty in x anyway: %0.3f > 0" % g.h
    zs = g.transform(xs, ys)
    print zip(xs, ys, zs)
    assert np.all(zs == 0), "z=0 bc there is no signal."
    predict_xs = g.predict(ys, zs)
    assert np.all(xs == predict_xs), predict_xs

def test_perfect():
    xs = np.repeat([0, 0, 1, 1, 1, 0], 5)
    ys = np.repeat([0, 0, 0, 1, 1, 1], 5)
    g = re.Remainder(xs, ys, k_max=2)
    zs = g.transform(xs, ys)
    print zip(xs, ys, zs)
    assert np.allclose(g.mi, 0, atol=0.01), "%0.3f > 0" % g.mi
    assert np.allclose(g.h, 0, atol=0.01), "%0.3f > 0" % g.h
    predict_xs = g.predict(ys, zs)
    assert np.all(xs == predict_xs), predict_xs

def test_mi():
    xs = [0, 0, 0, 0, 0, 0]
    ys = [0, 0, 0, 0, 0, 0]
    g = re.Remainder(xs, ys, k_max=2)
    g.pz_xy = np.array([[[1, 0], [0, 0]], [[0, 0], [0, 1]]])
    g.pxy = np.array([[0.5, 0], [0, 0.5]])
    assert np.allclose(g.mi, np.log(2)), "%0.3f > log(2)" % g.mi
    assert np.allclose(g.h, 0, atol=0.01), "%0.3f > 0" % g.h

def test_probabilistic():
    xs = [0, 0, 0, 1, 1, 0]
    ys = [0, 0, 0, 1, 1, 1]
    g = re.Remainder(xs, ys, k_max=3)
    zs = g.transform(xs, ys)
    print zip(xs, ys, zs)
    print g.pz_xy
    assert g.mi < 0.0001, "%0.5f" % g.mi
    assert g.h < 0.0001, "%0.5f" % g.h
    predict_xs = g.predict(ys, zs)
    assert np.all(xs == predict_xs), predict_xs

def test_deterministic():
    xs = [0, 0, 1, 1]
    ys = [0, 0, 1, 1]
    g = re.Remainder(xs, ys, k_max=2)
    zs = g.transform(xs, ys)
    print zip(xs, ys, zs)
    assert np.allclose(g.mi, 0, atol=0.01), "%0.3f > 0" % g.mi
    assert np.allclose(g.h, 0, atol=0.01), "%0.3f > 0" % g.h
    predict_xs = g.predict(ys, zs)
    assert np.all(xs == predict_xs), predict_xs
    assert np.all(g.transform([0]*100, [0]*100) == g.transform([0], [0]))

def test_identity():
    xs = np.array([0, 1, 0, 1])
    ys = np.array([0, 0, 1, 1])
    for _ in range(20):
        g = re.Remainder(xs, ys, k_max=2)
        zs = g.transform(xs, ys)
        assert np.all(zs == xs)
        predict_xs = g.predict(ys, zs)
        assert np.all(xs == predict_xs), predict_xs

def test_invertibility():
    xs = np.random.randint(0, 5, 100)
    ys = xs / 2 + np.random.randint(0, 2, 100)
    g = re.Remainder(xs, ys, k_max=8)
    zs = g.transform(xs, ys)
    print 'mi, h', g.mi, g.h
    print zip(xs, ys, zs)
    print np.array_str(g.pz_xy, precision=2, suppress_small=True)
    predict_xs = g.predict(ys, zs)
    print zip(predict_xs, xs)
    assert np.all(xs == predict_xs), predict_xs
    assert g.h < 0.01, "%0.5f" % g.h
    assert g.mi < 0.1, "%0.5f" % g.mi

def test_optimize():
    # Check whether the jacobian is approximately correct by comparing to finite difference
    from scipy.optimize import check_grad
    out = re.Remainder([0,0,1,1,2,2], [0,1,0,0,1,1], k_max=2)
    error = check_grad(out.objective, out.objective_jac, np.ravel(re.normalize_z(np.random.random(out.shape))))
    assert error < 1e-5, error