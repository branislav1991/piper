# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

import pytest

import jax
import jax.numpy as jnp

import piper
import piper.functional as func
from piper.distributions import normal
from piper import param


def test_kl_normal_normal_one_dimensional():
    model = piper.create_graph()
    model = normal(model, 'n1', jnp.array([0.]), jnp.array([1.]))
    model = normal(model, 'n2', jnp.array([0.]), jnp.array([1.]))
    model = normal(model, 'n3', jnp.array([1.]), jnp.array([1.]))
    model = normal(model, 'n4', jnp.array([1.]), jnp.array([2.]))

    assert func.kl_divergence(model, 'n1', 'n2') < 1e-6
    assert func.kl_divergence(model, 'n1', 'n3') - jnp.array(0.5) \
        < jnp.array(1e-6)
    assert func.kl_divergence(model, 'n3', 'n4') \
        - jnp.array(0.09657359028) < jnp.array(1e-6)


def test_kl_normal_normal_multi_dimensional():
    model = piper.create_graph()
    model = normal(model, 'n1', jnp.array([0., 0.]), jnp.array([1., 1.]))
    model = normal(model, 'n2', jnp.array([0., 0.]), jnp.array([1., 1.]))
    model = normal(model, 'n3', jnp.array([1., 1.]), jnp.array([1., 1.]))
    model = normal(model, 'n4', jnp.array([1., 1.]), jnp.array([2., 2.]))

    assert jnp.all(func.kl_divergence(model, 'n1', 'n2') < 1e-6)
    assert jnp.all(func.kl_divergence(model, 'n1', 'n3') - 0.5 < 1e-6)
    assert jnp.all(func.kl_divergence(model, 'n3', 'n4') - 0.09657359 < 1e-6)


def test_sample_normal():
    model = piper.create_graph()
    model = normal(model, 'n', jnp.array([0.]), jnp.array([1.]))

    keys = jax.random.split(jax.random.PRNGKey(123), 500)
    samples = jax.vmap(lambda k, m: func.sample(m, k)['n'].value,
                       in_axes=(0, None),
                       out_axes=0)(keys, model)

    assert abs(jnp.mean(samples)) < 0.2

    model = piper.create_graph()
    model = normal(model, 'n', jnp.array([10.]), jnp.array([1.]))

    keys = jax.random.split(jax.random.PRNGKey(123), 500)
    samples = jax.vmap(lambda k, m: func.sample(m, k)['n'].value,
                       in_axes=(0, None),
                       out_axes=0)(keys, model)

    assert abs(jnp.mean(samples)) - 10. < 0.2


def test_sample_normal_flexible_one_node():
    # test with one node
    model = piper.create_graph()
    model = normal(model, 'n', jnp.zeros((10, 10), dtype=jnp.float32),
                   param.flexible_param(jnp.array(1.0)))

    key = jax.random.PRNGKey(123)
    sample = func.sample(model, key)['n'].value
    assert sample.shape == (10, 10)


def test_sample_normal_flexible_joint():
    # test with joint model
    model = piper.create_graph()
    model = normal(model, 'weight', jnp.zeros((10, 10), dtype=jnp.float32),
                   jnp.zeros((10, 10), dtype=jnp.float32))
    model = normal(model, 'measurement', 'weight',
                   param.flexible_param(jnp.array(1.0)))

    key = jax.random.PRNGKey(123)
    sample = func.sample(model, key)['measurement'].value
    assert sample.shape == (10, 10)


def test_sample_joint_normal():
    model = piper.create_graph()
    model = normal(model, 'weight', jnp.array([0.]), jnp.array([1.]))
    model = normal(model, 'measurement', 'weight', jnp.array([1.]))

    keys = jax.random.split(jax.random.PRNGKey(123), 500)
    samples = jax.vmap(lambda k, m: func.sample(m, k)['measurement'].value,
                       in_axes=(0, None),
                       out_axes=0)(keys, model)

    assert abs(jnp.mean(samples)) < 0.3


def test_incompatible_dimensions():
    model = piper.create_graph()
    model = normal(model, 'weight', jnp.array([0., 0.]), jnp.array([1., 1.]))
    model = normal(model, 'measurement', 'weight', jnp.array([1.]))

    with pytest.raises(RuntimeError):
        key = jax.random.PRNGKey(123)
        func.sample(model, key)


def test_sample_conditioned():
    model = piper.create_graph()
    model = normal(model, 'weight', jnp.array([0.]), jnp.array([1.]))
    model = normal(model, 'measurement', 'weight', jnp.array([1.]))
    model = func.condition(model, 'weight', jnp.array([0.]))

    keys = jax.random.split(jax.random.PRNGKey(123), 500)
    samples = jax.vmap(lambda k, m: func.sample(m, k)['measurement'].value,
                       in_axes=(0, None),
                       out_axes=0)(keys, model)

    assert abs(jnp.mean(samples)) < 0.2


def test_sample_conditioned_posterior_error():
    """This test should return an error as direct sampling from
    posterior is not possible.
    """
    model = piper.create_graph()
    model = normal(model, 'n1', jnp.array([0.]), jnp.array([1.]))
    model = normal(model, 'n2', 'n1', jnp.array([1.]))
    model = func.condition(model, 'n2', jnp.array([0.]))

    key = jax.random.PRNGKey(123)
    with pytest.raises(RuntimeError):
        func.sample(model, key)

    model = piper.create_graph()
    model = normal(model, 'n1', jnp.array([0.]), jnp.array([1.]))
    model = normal(model, 'n2', 'n1', jnp.array([1.]))
    model = normal(model, 'n3', 'n2', jnp.array([1.]))
    model = func.condition(model, 'n3', jnp.array([0.]))

    key = jax.random.PRNGKey(123)
    with pytest.raises(RuntimeError):
        func.sample(model, key)
