# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

import pytest

import jax.numpy as jnp

from piper.distributions import normal


def test_sample_normal():
    with pytest.raises(AssertionError):
        normal.Normal(0, 1)

    fixture = normal.Normal(jnp.array(0.), jnp.array(1.))
    samples = []
    for i in range(100):
        samples.append(fixture.sample())

    samples = jnp.stack(samples)
    assert abs(jnp.mean(samples)) < 0.1
