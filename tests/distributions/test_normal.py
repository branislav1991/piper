# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

import pytest

import jax.numpy as jnp

import piper
from piper.distributions import normal
from piper.functional import kl_divergence


def test_throw_bad_params():
    # Should fail because of missing model
    with pytest.raises(ValueError):
        normal(None, 'test', jnp.array([0.]), jnp.array([1.]))

    # Should fail because of wrong parameters
    model = piper.create_graph()
    with pytest.raises(TypeError):
        normal(model, 'test2', 0., 1.)

    # Should fail because of duplicate entries in the graph
    normal(model, 'test3', jnp.array([0.]), jnp.array([1.]))
    with pytest.raises(ValueError):
        normal(model, 'test3', jnp.array([0.]), jnp.array([1.]))

    # Should fail because of different shapes of mu and sigma
    with pytest.raises(ValueError):
        normal(model, 'test3', jnp.array([0., 0.]), jnp.array([1.]))


def test_sample_normal():
    model = piper.create_graph()
    model = normal(model, 'weight', jnp.array([0.]), jnp.array([1.]))
    samples = []
    for i in range(100):
        samples.append(piper.sample(model)['weight'])

    samples = jnp.stack(samples)
    assert abs(jnp.mean(samples)) < 0.1


def test_kl_normal_normal_one_dimensional():
    model = piper.create_graph()
    model = normal(model, 'n1', jnp.array([0.]), jnp.array([1.]))
    model = normal(model, 'n2', jnp.array([0.]), jnp.array([1.]))
    model = normal(model, 'n3', jnp.array([1.]), jnp.array([1.]))
    model = normal(model, 'n4', jnp.array([1.]), jnp.array([2.]))

    assert kl_divergence.kl_divergence(model, 'n1', 'n2') < 1e-6
    assert kl_divergence.kl_divergence(model, 'n1', 'n3') - jnp.array(0.5) \
        < jnp.array(1e-6)
    assert kl_divergence.kl_divergence(model, 'n3', 'n4') \
        - jnp.array(0.09657359028) < jnp.array(1e-6)


def test_kl_normal_normal_multi_dimensional():
    model = piper.create_graph()
    model = normal(model, 'n1', jnp.array([0., 0.]), jnp.array([1., 1.]))
    model = normal(model, 'n2', jnp.array([0., 0.]), jnp.array([1., 1.]))
    model = normal(model, 'n3', jnp.array([1., 1.]), jnp.array([1., 1.]))
    model = normal(model, 'n4', jnp.array([1., 1.]), jnp.array([2., 2.]))

    assert jnp.all(kl_divergence.kl_divergence(model, 'n1', 'n2') < 1e-6)
    assert jnp.all(kl_divergence.kl_divergence(model, 'n1', 'n3') - 0.5
                   < 1e-6)
    assert jnp.all(kl_divergence.kl_divergence(model, 'n3', 'n4')
                   - 0.09657359 < 1e-6)
