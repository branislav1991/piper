# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

from piper import core
from piper.distributions.distribution import Distribution


def compute_kl_div(dist1: Distribution, dist2: Distribution):
    if (type(dist1), type(dist2)) not in core._KL_REGISTRY:
        raise ValueError(
            f"KL-divergence between {type(dist1)} and {type(dist2)} \
             not defined")

    fn = core._KL_REGISTRY[(type(dist1), type(dist2))]
    return fn(dist1, dist2)
