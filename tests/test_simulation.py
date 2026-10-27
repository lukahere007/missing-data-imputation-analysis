import numpy as np
import pytest

import simulation as sim


def test_true_coefficients():
    np.testing.assert_allclose(sim.true_coefficients(), [0.25, 0.25])


@pytest.mark.parametrize("mechanism", sim.MECHANISMS)
def test_missing_fraction(mechanism):
    dat = sim.sim_dat(400, np.random.default_rng(0))
    mask = sim.missing_mask(dat, mechanism, p=0.3, rng=np.random.default_rng(1))
    assert mask.sum() == 120


def test_mar_on_y_selects_high_y():
    dat = sim.sim_dat(2000, np.random.default_rng(0))
    mask = sim.missing_mask(dat, "MAR (y)", p=0.3, rng=np.random.default_rng(1))
    assert dat.y[mask].mean() > dat.y[~mask].mean() + 0.5


def test_ols_matches_lstsq():
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(100), rng.normal(size=(100, 2))])
    y = X @ [1.0, 2.0, -1.0] + rng.normal(size=100)
    b, se, df = sim.ols(X, y)
    np.testing.assert_allclose(b, np.linalg.lstsq(X, y, rcond=None)[0])
    assert df == 97 and np.all(se > 0)


def test_rubin_pool_without_between_variance():
    est = np.tile([[0.5, 1.0]], (5, 1))
    var = np.tile([[0.04, 0.09]], (5, 1))
    q, se, _ = sim.rubin_pool(est, var, df_complete=100)
    np.testing.assert_allclose(q, [0.5, 1.0])
    np.testing.assert_allclose(se, [0.2, 0.3])


def test_rubin_pool_adds_between_variance():
    est = np.array([[0.0], [1.0], [2.0]])
    var = np.full((3, 1), 1.0)
    q, se, df = sim.rubin_pool(est, var, df_complete=1000)
    np.testing.assert_allclose(q, [1.0])
    np.testing.assert_allclose(se**2, [1.0 + (1 + 1 / 3) * 1.0])
    assert 0 < df[0] < 1000


def test_imputations_only_change_missing_rows():
    rng = np.random.default_rng(3)
    dat = sim.sim_dat(300, rng)
    mask = sim.missing_mask(dat, "MCAR", 0.3, rng)
    for filled in sim.draw_imputations(dat, mask, m=3, rng=rng):
        np.testing.assert_array_equal(filled[~mask], dat.x1.values[~mask])
        assert np.isfinite(filled).all()


def test_small_monte_carlo_runs():
    res = sim.monte_carlo(n_reps=3, n=200, m=5, seed=1)
    summ = sim.summarise(res)
    assert len(summ) == 2 * len(sim.MECHANISMS) * (len(sim.METHODS) + 1)
    assert summ.coverage.between(0, 1).all()
