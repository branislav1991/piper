# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

import jax.numpy as jnp

_default_tolerance = {
    jnp.dtype(jnp.bool_): 0,
    jnp.dtype(jnp.int8): 0,
    jnp.dtype(jnp.int16): 0,
    jnp.dtype(jnp.int32): 0,
    jnp.dtype(jnp.int64): 0,
    jnp.dtype(jnp.uint8): 0,
    jnp.dtype(jnp.uint16): 0,
    jnp.dtype(jnp.uint32): 0,
    jnp.dtype(jnp.uint64): 0,
    jnp.dtype(jnp.bfloat16): 1e-2,
    jnp.dtype(jnp.float16): 1e-2,
    jnp.dtype(jnp.float32): 1e-3,
    jnp.dtype(jnp.float64): 1e-10,
    jnp.dtype(jnp.complex64): 1e-3,
    jnp.dtype(jnp.complex128): 1e-10,
}


def tolerance(dtype, tol=None):
    if not tol:
        return _default_tolerance[dtype]
    else:
        return tol


def assert_allclose(a, b, atol=None, rtol=None):
    kw = {}
    if atol:
        kw["atol"] = atol
    if rtol:
        kw["rtol"] = rtol
    assert jnp.allclose(a, b, **kw)


def check_close(a, b, atol=None, rtol=None):
    if not isinstance(a, jnp.ndarray):
        a = jnp.array(a)

    if not isinstance(b, jnp.ndarray):
        b = jnp.array(b)

    assert a.shape == b.shape
    atol = max(tolerance(a.dtype, atol), tolerance(b.dtype, atol))
    rtol = max(tolerance(a.dtype, rtol), tolerance(b.dtype, rtol))
    assert_allclose(a, b, atol=atol * a.size, rtol=rtol * b.size)
