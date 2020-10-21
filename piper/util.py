# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

import jax.random
import jax.numpy as jnp

default_key = jax.random.PRNGKey(666)


def split_default_key() -> jnp.ndarray:
    """Get a new rng key.

    Returns:
        Newly generated rng key.
    """
    global default_key
    default_key, newkey = jax.random.split(default_key)
    return newkey


def set_rng_seed(seed: int):
    """Set default rng seed.

    Args:
        seed: A 64- or 32-bit integer used as the value of the key.
    """
    global default_key
    default_key = jax.random.PRNGKey(seed)
