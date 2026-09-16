"""Fuzzy membership of spatial units in regions, derived from interaction data.

The core idea is a proportional membership function: a unit's membership in a
region is the share of its total interaction that is directed at that region.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "interaction_membership",
    "aggregate_interactions",
    "region_membership",
    "combine",
]


def _matrix_and_labels(interactions, labels, symmetrize):
    """Validate the pair and return (matrix, index, labels, indicator, regions)."""
    if isinstance(interactions, pd.DataFrame):
        if not interactions.index.equals(interactions.columns):
            raise ValueError("interaction DataFrame must have identical index and columns")
        matrix, index = interactions.to_numpy(dtype=float), interactions.index
    else:
        matrix = np.asarray(interactions, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"interactions must be square, got shape {matrix.shape}")
        index = pd.RangeIndex(matrix.shape[0])

    labels = np.asarray(labels)
    if len(labels) != matrix.shape[0]:
        raise ValueError(
            f"labels has length {len(labels)} but interactions has {matrix.shape[0]} units"
        )
    if pd.isna(labels).any():
        raise ValueError(
            f"labels contain {int(pd.isna(labels).sum())} missing value(s); every unit must "
            "be assigned to a region. Areas outside the plan need dropping before you start."
        )

    if symmetrize:
        matrix = matrix + matrix.T

    regions = np.unique(labels)
    indicator = (labels[:, None] == regions[None, :]).astype(float)
    return matrix, index, indicator, regions


def _row_normalize(values):
    """Scale each row to sum to one, leaving all-zero rows as zeros."""
    totals = values.sum(axis=1, keepdims=True)
    return np.divide(values, totals, out=np.zeros_like(values), where=totals != 0)


def interaction_membership(interactions, labels, *, symmetrize=False):
    """Fuzzy membership of every unit in every region.

    Implements

    .. math::

        \\mu_{D_j}(x) = \\frac{\\sum_{y \\in D_j} I_{xy}}
                              {\\sum_{k} \\sum_{y \\in D_k} I_{xy}}

    the share of unit ``x``'s total interaction that is directed at region
    ``D_j``. Each row of the result sums to one.

    Parameters
    ----------
    interactions : ndarray or DataFrame of shape (n, n)
        Interaction strength between units, e.g. mobility flows or estimated
        social connections. ``interactions[i, j]`` is the interaction from unit
        ``i`` to unit ``j``.
    labels : array-like of shape (n,)
        Region each unit is assigned to under the plan being evaluated.
    symmetrize : bool, default False
        Treat a directed interaction matrix as undirected by adding its
        transpose. Use this for origin-destination flows; leave it off for data
        that is already symmetric, such as social connectedness.

    Returns
    -------
    DataFrame of shape (n, k)
        Memberships, indexed by unit and with one column per region.
    """
    matrix, index, indicator, regions = _matrix_and_labels(interactions, labels, symmetrize)
    # (n, n) @ (n, k) -> (n, k): each unit's total interaction with each region.
    return pd.DataFrame(_row_normalize(matrix @ indicator), index=index, columns=regions)


def aggregate_interactions(interactions, labels, *, symmetrize=False):
    """Collapse a unit-by-unit interaction matrix to region-by-region totals.

    Returns a ``k x k`` DataFrame where entry ``(i, j)`` is the total
    interaction between the units of region ``i`` and those of region ``j``.
    """
    matrix, _, indicator, regions = _matrix_and_labels(interactions, labels, symmetrize)
    return pd.DataFrame(indicator.T @ matrix @ indicator, index=regions, columns=regions)


def region_membership(interactions, labels, *, symmetrize=False):
    """Fuzzy membership of each region in every region.

    The region-level analogue of :func:`interaction_membership`: units are
    aggregated into their regions first, so the result describes how each
    region as a whole divides its interaction among the regions.
    """
    totals = aggregate_interactions(interactions, labels, symmetrize=symmetrize)
    return pd.DataFrame(
        _row_normalize(totals.to_numpy()), index=totals.index, columns=totals.columns
    )


def combine(*memberships, method="product"):
    """Fuse membership sets derived from different kinds of interaction.

    Parameters
    ----------
    *memberships : DataFrame
        Two or more membership tables sharing a set of region columns. Only
        units present in every table are kept, since a unit missing from one
        source has no combined view.
    method : {"product", "mean"}, default "product"
        ``"product"`` multiplies the memberships element-wise and renormalizes.
        Agreement between sources amplifies a region's membership and
        disagreement suppresses it, which sharpens the distribution.
        ``"mean"`` takes the arithmetic mean instead, which averages the
        perspectives without sharpening them.

    Returns
    -------
    DataFrame
        Combined memberships, rows summing to one.
    """
    if len(memberships) < 2:
        raise ValueError("combine() needs at least two membership tables")

    columns = memberships[0].columns
    for i, table in enumerate(memberships[1:], start=1):
        if not table.columns.equals(columns):
            raise ValueError(f"membership table {i} has different region columns than table 0")

    index = memberships[0].index
    for table in memberships[1:]:
        index = index.intersection(table.index)
    aligned = [table.loc[index].to_numpy(dtype=float) for table in memberships]

    if method == "product":
        stacked = np.prod(aligned, axis=0)
    elif method == "mean":
        stacked = np.mean(aligned, axis=0)
    else:
        raise ValueError(f"unknown method {method!r}, expected 'product' or 'mean'")

    return pd.DataFrame(_row_normalize(stacked), index=index, columns=columns)
