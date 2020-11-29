# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

import jax.numpy as jnp


def is_floating(val: jnp.ndarray):
    return val.dtype in [jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64]


def is_integer(val: jnp.ndarray):
    int_type = val.dtype in [jnp.int32, jnp.int64]
    whole_number = jnp.equal(jnp.mod(val, 1), 0)
    return jnp.logical_or(int_type, whole_number)
