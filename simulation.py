"""Simulation helpers for the missing-data imputation study.

Data-generating process (unchanged from the original notebook):
    (y, x1, x2) ~ N(0, Sigma), Sigma = 1.5 on the diagonal, 0.5 off the diagonal.
The population regression of y on x1 and x2 then has slopes beta1 = beta2 = 0.25.

Missingness is introduced in the covariate x1 under four mechanisms, and four
methods are compared: complete-case analysis, mean imputation, deterministic
regression imputation, and multiple imputation pooled with Rubin's rules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

MECHANISMS = ("MCAR", "MAR (x2)", "MAR (y)", "MNAR (x1)")
METHODS = ("Complete case", "Mean imputation", "Regression imputation", "Multiple imputation")

# What drives missingness of x1 under each mechanism (None = pure chance).
_DRIVER = {"MCAR": None, "MAR (x2)": "x2", "MAR (y)": "y", "MNAR (x1)": "x1"}


def covariance(npred: int = 2) -> np.ndarray:
    sigma = np.full((npred + 1, npred + 1), 0.5)
    np.fill_diagonal(sigma, 1.5)
    return sigma


def true_coefficients(npred: int = 2) -> np.ndarray:
    """Population slopes of y on x1..xk implied by the covariance matrix."""
    s = covariance(npred)
    return np.linalg.solve(s[1:, 1:], s[1:, 0])


def sim_dat(n: int = 500, rng: np.random.Generator | None = None, npred: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(rng)
    dat = rng.multivariate_normal(np.zeros(npred + 1), covariance(npred), size=n)
    return pd.DataFrame(dat, columns=["y"] + [f"x{i}" for i in range(1, npred + 1)])


def missing_mask(dat: pd.DataFrame, mechanism: str, p: float = 0.3,
                 rng: np.random.Generator | None = None, noise_sd: float = 1.0) -> np.ndarray:
    """Boolean mask of rows where x1 is set to missing.

    Exactly round(p * n) rows are removed. For MAR/MNAR the rows with the largest
    value of (standardised driver + N(0, noise_sd^2)) are removed, so high values of
    the driver are more likely to be missing (the same ranking idea as the original
    notebook's sim_mar).
    """
    rng = np.random.default_rng(rng)
    n = len(dat)
    k = int(round(p * n))
    driver = _DRIVER[mechanism]
    if driver is None:
        score = rng.random(n)
    else:
        z = dat[driver].to_numpy()
        score = (z - z.mean()) / z.std() + rng.normal(0.0, noise_sd, n)
    mask = np.zeros(n, dtype=bool)
    mask[np.argsort(score)[-k:]] = True
    return mask


# --------------------------------------------------------------------------- OLS

def ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """OLS with intercept already in X. Returns (beta, standard errors, residual df)."""
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    df = X.shape[0] - X.shape[1]
    sigma2 = resid @ resid / df
    return beta, np.sqrt(np.diag(XtX_inv) * sigma2), df


def _design(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones_like(x1), x1, x2])


# ----------------------------------------------------------------------- methods
# Each method returns (estimate, standard error, degrees of freedom) for every
# coefficient [intercept, x1, x2] of the analysis model y ~ x1 + x2.

def complete_case(dat, mask):
    obs = ~mask
    return ols(_design(dat.x1.values[obs], dat.x2.values[obs]), dat.y.values[obs])


def mean_imputation(dat, mask):
    x1 = dat.x1.values.copy()
    x1[mask] = x1[~mask].mean()
    return ols(_design(x1, dat.x2.values), dat.y.values)


def regression_imputation(dat, mask):
    """Single, deterministic imputation: fitted value of x1 ~ x2 + y (no noise added)."""
    x1 = dat.x1.values.copy()
    Z = np.column_stack([np.ones(len(dat)), dat.x2.values, dat.y.values])
    gamma, _, _ = ols(Z[~mask], x1[~mask])
    x1[mask] = Z[mask] @ gamma
    return ols(_design(x1, dat.x2.values), dat.y.values)


def draw_imputations(dat, mask, m: int = 20, rng: np.random.Generator | None = None) -> list[np.ndarray]:
    """m proper Bayesian draws of x1 under a normal linear model x1 ~ x2 + y.

    For each imputation: draw sigma^2 from its posterior (scaled inverse chi-square),
    draw the regression coefficients given sigma^2, then add residual noise to the
    predictions. Drawing the parameters is what makes the imputations "proper"
    (Rubin 1987; van Buuren 2018, algorithm 3.1). This is what a MICE / chained
    equations routine does when only one variable is incomplete.
    """
    rng = np.random.default_rng(rng)
    x1 = dat.x1.values
    Z = np.column_stack([np.ones(len(dat)), dat.x2.values, dat.y.values])
    Zo, xo = Z[~mask], x1[~mask]
    ZtZ_inv = np.linalg.inv(Zo.T @ Zo)
    gamma_hat = ZtZ_inv @ Zo.T @ xo
    resid = xo - Zo @ gamma_hat
    df = len(xo) - Z.shape[1]
    chol = np.linalg.cholesky(ZtZ_inv)
    out = []
    for _ in range(m):
        sigma2 = resid @ resid / rng.chisquare(df)
        gamma = gamma_hat + np.sqrt(sigma2) * chol @ rng.standard_normal(Z.shape[1])
        filled = x1.copy()
        filled[mask] = Z[mask] @ gamma + rng.normal(0.0, np.sqrt(sigma2), mask.sum())
        out.append(filled)
    return out


def rubin_pool(estimates: np.ndarray, variances: np.ndarray, df_complete: float):
    """Pool m sets of estimates with Rubin's rules.

    Uses the Barnard & Rubin (1999) small-sample degrees of freedom.
    Returns (pooled estimate, pooled SE, degrees of freedom).
    """
    m = estimates.shape[0]
    qbar = estimates.mean(axis=0)
    W = variances.mean(axis=0)
    B = estimates.var(axis=0, ddof=1)
    T = W + (1 + 1 / m) * B
    lam = np.clip((1 + 1 / m) * B / T, 1e-12, 1.0)
    df_old = (m - 1) / lam**2
    df_obs = (df_complete + 1) / (df_complete + 3) * df_complete * (1 - lam)
    df = df_old * df_obs / (df_old + df_obs)
    return qbar, np.sqrt(T), df


def multiple_imputation(dat, mask, m: int = 20, rng: np.random.Generator | None = None):
    ests, vars_ = [], []
    df_c = len(dat) - 3
    for x1 in draw_imputations(dat, mask, m=m, rng=rng):
        b, se, df_c = ols(_design(x1, dat.x2.values), dat.y.values)
        ests.append(b)
        vars_.append(se**2)
    return rubin_pool(np.array(ests), np.array(vars_), df_c)


def fit_all(dat, mask, m: int = 20, rng=None) -> dict[str, tuple]:
    return {
        "Complete case": complete_case(dat, mask),
        "Mean imputation": mean_imputation(dat, mask),
        "Regression imputation": regression_imputation(dat, mask),
        "Multiple imputation": multiple_imputation(dat, mask, m=m, rng=rng),
    }


# ------------------------------------------------------------------- Monte Carlo

def monte_carlo(n_reps: int = 500, n: int = 500, p: float = 0.3, m: int = 20,
                seed: int = 2024) -> pd.DataFrame:
    """Replicate the full experiment. One row per replication x mechanism x method,
    for the x1 and x2 slopes, with the estimate, SE and 95% CI coverage."""
    beta = true_coefficients()
    rows = []
    for rep, child in enumerate(np.random.SeedSequence(seed).spawn(n_reps)):
        rng = np.random.default_rng(child)
        dat = sim_dat(n, rng)
        full = ols(_design(dat.x1.values, dat.x2.values), dat.y.values)
        for mech in MECHANISMS:
            mask = missing_mask(dat, mech, p, rng)
            fits = {"Full data (no missingness)": full, **fit_all(dat, mask, m=m, rng=rng)}
            for method, (b, se, df) in fits.items():
                df = np.broadcast_to(df, b.shape)
                for j, coef in ((1, "x1"), (2, "x2")):
                    half = stats.t.ppf(0.975, df[j]) * se[j]
                    rows.append((rep, mech, method, coef, b[j], se[j],
                                 abs(b[j] - beta[j - 1]) <= half))
    return pd.DataFrame(rows, columns=["rep", "mechanism", "method", "coef",
                                       "estimate", "se", "covered"])


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    beta = dict(zip(["x1", "x2"], true_coefficients()))
    g = results.groupby(["coef", "mechanism", "method"], sort=False)
    out = g.agg(mean_estimate=("estimate", "mean"), empirical_sd=("estimate", "std"),
                mean_se=("se", "mean"), coverage=("covered", "mean"), reps=("rep", "size"))
    truth = out.index.get_level_values("coef").map(beta)
    out.insert(1, "bias", out["mean_estimate"] - truth)
    out["rmse"] = np.sqrt(out["bias"] ** 2 + out["empirical_sd"] ** 2)
    return out.reset_index()
