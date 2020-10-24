# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

import pytest

import jax.numpy as jnp

import piper
from piper.distributions import normal


def test_throw_bad_params():
    # Should fail because of missing model
    with pytest.raises(ValueError):
        normal(None, 'test', jnp.array(0.), jnp.array(1.))

    # Should fail because of wrong parameters
    model = piper.create_graph()
    with pytest.raises(TypeError):
        normal(model, 'test2', 0., 1.)

    # Should fail because of duplicate entries in the graph
    normal(model, 'test3', jnp.array(0.), jnp.array(1.))
    with pytest.raises(ValueError):
        normal(model, 'test3', jnp.array(0.), jnp.array(1.))


def test_sample_normal():
    model = piper.create_graph()
    model = normal(model, 'weight', jnp.array(0.), jnp.array(1.))
    samples = []
    for i in range(100):
        samples.append(piper.sample(model)['weight'])

    samples = jnp.stack(samples)
    assert abs(jnp.mean(samples)) < 0.1
