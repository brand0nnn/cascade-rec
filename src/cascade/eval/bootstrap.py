"""Paired bootstrap over users for the difference between two systems (CLAUDE.md §5)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def paired_bootstrap(users: np.ndarray, metric_a: np.ndarray, metric_b: np.ndarray, n_resamples: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict[str, float]:
    """Macro (per-user) difference mean(B) - mean(A) with a percentile CI from resampling users.

    ``metric_a`` / ``metric_b`` are per-instance values over the *same* instances (aligned), ``users`` the owner of each.
    """
    df = pd.DataFrame({"u": users, "d": np.asarray(metric_b, float) - np.asarray(metric_a, float)})
    per_user = df.groupby("u")["d"].mean().values
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_user), size=(n_resamples, len(per_user)))
    boots = per_user[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"diff": float(per_user.mean()), "ci_low": float(lo), "ci_high": float(hi), "n_users": int(len(per_user)), "n_resamples": n_resamples}
