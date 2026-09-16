"""Information-theoretic summaries of fuzzy memberships.

Entropy measures how divided a unit's membership is across regions;
information gain measures how much that division falls when two perspectives
are combined; KL divergence compares those distributions region by region.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "entropy",
    "information_gain",
    "kl_divergence",
    "kl_divergence_by_region",
]


def _xlogy(x, y):
    """x * log(y), taking 0 * log(0) to be 0."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out = np.zeros(np.broadcast(x, y).shape, dtype=float)
    nonzero = x > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        out[nonzero] = x[nonzero] * np.log(np.broadcast_to(y, out.shape)[nonzero])
    return out


def entropy(memberships, base=2):
    """Shannon entropy of each unit's membership distribution.

    .. math:: H(x) = -\\sum_j \\mu_{D_j}(x) \\log \\mu_{D_j}(x)

    Zero memberships contribute nothing, following the convention
    :math:`0 \\log 0 = 0`. Entropy runs from 0, when a unit sits entirely in one
    region, up to :math:`\\log_{base} k`, when its membership is spread evenly
    across all ``k`` regions.

    Parameters
    ----------
    memberships : DataFrame of shape (n, k)
    base : float, default 2
        Logarithm base. 2 gives entropy in bits.

    Returns
    -------
    Series of length n
    """
    values = memberships.to_numpy(dtype=float)
    result = -_xlogy(values, values).sum(axis=1) / np.log(base)
    return pd.Series(result, index=memberships.index, name="entropy")


def information_gain(source, combined, base=2):
    """Entropy reduction from using the combined memberships instead of `source`.

    .. math:: \\text{IG}(x) = H^{\\text{source}}(x) - H^{\\text{combined}}(x)

    Positive values mean combining sharpened the unit's membership; negative
    values mean it became more dispersed. Only units present in both tables are
    returned.
    """
    source_h, combined_h = entropy(source, base=base), entropy(combined, base=base)
    shared = source_h.index.intersection(combined_h.index)
    return (source_h.loc[shared] - combined_h.loc[shared]).rename("information_gain")


def kl_divergence(p, q):
    """KL divergence of distribution `q` from `p`, in nats.

    Both arguments are normalized to sum to one first. Returns ``inf`` where
    ``p`` puts mass on an outcome ``q`` rules out.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != q.shape:
        raise ValueError(f"p and q must have the same shape, got {p.shape} and {q.shape}")
    p_sum, q_sum = p.sum(), q.sum()
    if p_sum == 0 or q_sum == 0:
        raise ValueError("p and q must each contain positive mass")
    p, q = p / p_sum, q / q_sum
    if np.any((p > 0) & (q == 0)):
        return np.inf
    return float((_xlogy(p, p) - _xlogy(p, q)).sum())


def kl_divergence_by_region(source_values, combined_values, labels):
    """KL divergence between two per-unit value distributions, within each region.

    Used in the paper to compare per-unit entropy under one perspective against
    the combined perspective, region by region: the values of the units in a
    region are normalized into a distribution and the two are compared. A high
    divergence marks a region where that perspective is most distinct from the
    combined view.

    Parameters
    ----------
    source_values, combined_values : Series indexed by unit
        Per-unit quantities to compare, typically entropies.
    labels : Series indexed by unit
        Region label per unit.

    Returns
    -------
    Series indexed by region
    """
    frame = pd.DataFrame(
        {"source": source_values, "combined": combined_values, "region": labels}
    ).dropna()
    divergences = {
        region: kl_divergence(group["source"].to_numpy(), group["combined"].to_numpy())
        for region, group in frame.groupby("region", sort=True)
    }
    return pd.Series(divergences, name="kl_divergence").rename_axis("region")
