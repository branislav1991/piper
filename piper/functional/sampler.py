# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

import jax.numpy as jnp

from piper import core
from piper.distributions import distribution as dist


def sample(sample_name: str, d: dist.Distribution, key: jnp.ndarray):
    sample = None
    for mod in core._MODIFIER_STACK[::-1]:
        new_sample = mod.process(sample_name, d)
        sample = sample if new_sample is None else new_sample

    if sample is None:
        sample = d.sample(key)

    for mod in core._MODIFIER_STACK[::-1]:
        new_sample = mod.post_process(sample, sample_name, d)
        sample = sample if new_sample is None else new_sample

    return sample
