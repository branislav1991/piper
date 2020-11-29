# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

import jax.numpy as jnp

from piper import core
from piper.distributions import distribution as dist


def sample(sample_name: str, d: dist.Distribution, key: jnp.ndarray):
    message = {
        'name': sample_name,
        'sample': None,
        'distribution': d,
        'is_conditioned': False
    }

    for mod in core._MODIFIER_STACK[::-1]:
        message = mod.process(message)

    if message['sample'] is None:
        message['sample'] = d.sample(key)

    for mod in core._MODIFIER_STACK[::-1]:
        message = mod.post_process(message)

    return message['sample']
