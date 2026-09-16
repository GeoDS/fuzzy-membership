"""Defuzzification: turning fuzzy memberships back into crisp judgements."""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["assign", "classify"]

ALIGNED = "aligned"
LOW_MEMBERSHIP = "low_membership"
MISASSIGNED = "misassigned"
NO_INTERACTION = "no_interaction"


def assign(memberships):
    """Assign each unit to the region it has the strongest membership in.

    .. math:: \\text{Region}(x) = \\arg\\max_j \\mu_{D_j}(x)

    Ties go to the first region in column order.
    """
    return memberships.idxmax(axis=1).rename("assigned_region")


def classify(memberships, labels, *, threshold=0.5):
    """Compare interaction-implied regions against the regions of a plan.

    Each unit falls into one of three categories:

    ``misassigned``
        Its strongest membership is in a region other than the one the plan
        puts it in.
    ``low_membership``
        The plan agrees with its strongest membership, but that membership is
        below `threshold` -- the unit is in the right region while holding only
        a weak affiliation with it.
    ``aligned``
        The plan agrees and the membership is at or above `threshold`.
    ``no_interaction``
        The unit has no recorded interaction at all, so its memberships are
        all zero and say nothing about where it belongs. Without this it would
        be counted as misassigned, since the strongest of several zeros is the
        first region in column order.

    Parameters
    ----------
    memberships : DataFrame of shape (n, k)
    labels : array-like or Series of length n
        Region each unit is assigned to under the plan.
    threshold : float, default 0.5
        Membership below which an otherwise correct placement counts as weak.

    Returns
    -------
    DataFrame indexed by unit, with columns ``plan_region``,
    ``assigned_region``, ``max_membership`` and ``status``.
    """
    if not 0 <= threshold <= 1:
        raise ValueError(f"threshold must be between 0 and 1, got {threshold}")
    if not isinstance(labels, pd.Series):
        labels = pd.Series(np.asarray(labels), index=memberships.index)
    labels = labels.reindex(memberships.index)

    best_region = assign(memberships)
    best_value = memberships.max(axis=1)

    agrees = best_region.to_numpy() == labels.to_numpy()
    status = np.where(
        ~agrees,
        MISASSIGNED,
        np.where(best_value.to_numpy() < threshold, LOW_MEMBERSHIP, ALIGNED),
    )
    # An all-zero row is an absence of evidence, not evidence of misassignment.
    status = np.where(memberships.sum(axis=1).to_numpy() == 0, NO_INTERACTION, status)
    return pd.DataFrame(
        {
            "plan_region": labels,
            "assigned_region": best_region,
            "max_membership": best_value,
            "status": status,
        },
        index=memberships.index,
    )
