"""Fuzzy membership of spatial units in regions, from interaction networks.

Measures how strongly each spatial unit belongs to each region of a plan,
using the interactions -- mobility flows, social ties -- that connect it to the
units around it. A unit's membership in a region is the share of its total
interaction directed at that region, so units far inside a region belong to it
almost entirely while units near a boundary divide their membership across
several.

Typical use::

    from fuzzy_membership import interaction_membership, combine, entropy

    spatial = interaction_membership(flows, districts, symmetrize=True)
    social = interaction_membership(sci, districts)
    combined = combine(spatial, social)
    divided = entropy(combined)
"""

from fuzzy_membership.assignment import assign, classify
from fuzzy_membership.membership import (
    combine,
    interaction_membership,
    region_membership,
)
from fuzzy_membership.metrics import (
    entropy,
    information_gain,
    kl_divergence_by_region,
)

__version__ = "0.1.0"

__all__ = [
    "assign",
    "classify",
    "combine",
    "entropy",
    "information_gain",
    "interaction_membership",
    "kl_divergence_by_region",
    "region_membership",
]
