# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

import jax.random
import jax.numpy as jnp

from piper import util


def test_rng_key():
    # Let us check if the default rng key is split properly

    sample_seed = 100
    key = jax.random.PRNGKey(sample_seed)
    util.set_rng_seed(sample_seed)

    _, newkey = jax.random.split(key)
    newkey_util = util.split_default_key()

    assert newkey.shape == (2,) and newkey.dtype == jnp.uint32
    assert newkey_util.shape == (2,) and newkey_util.dtype == jnp.uint32
    assert (newkey == newkey_util).all()
