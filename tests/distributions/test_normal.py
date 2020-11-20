# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

import pytest

import jax
import jax.numpy as jnp

import piper.functional as func
import piper.models as models
from piper.distributions import normal
from piper import param


def test_kl_normal_normal_one_dimensional():
    model = models.create_forward_model()
    model = normal(model, 'n1', jnp.array([0.]), jnp.array([1.]))
    model = normal(model, 'n2', jnp.array([0.]), jnp.array([1.]))
    model = normal(model, 'n3', jnp.array([1.]), jnp.array([1.]))
    model = normal(model, 'n4', jnp.array([1.]), jnp.array([2.]))

    assert func.compute_kl_div(model, 'n1', 'n2') < 1e-6
    assert func.compute_kl_div(model, 'n1', 'n3') - jnp.array(0.5) \
        < jnp.array(1e-6)
    assert func.compute_kl_div(model, 'n3', 'n4') \
        - jnp.array(0.09657359028) < jnp.array(1e-6)


def test_kl_normal_normal_multi_dimensional():
    model = models.create_forward_model()
    model = normal(model, 'n1', jnp.array([0., 0.]), jnp.array([1., 1.]))
    model = normal(model, 'n2', jnp.array([0., 0.]), jnp.array([1., 1.]))
    model = normal(model, 'n3', jnp.array([1., 1.]), jnp.array([1., 1.]))
    model = normal(model, 'n4', jnp.array([1., 1.]), jnp.array([2., 2.]))

    assert jnp.all(func.compute_kl_div(model, 'n1', 'n2') < 1e-6)
    assert jnp.all(func.compute_kl_div(model, 'n1', 'n3') - 0.5 < 1e-6)
    assert jnp.all(func.compute_kl_div(model, 'n3', 'n4') - 0.09657359 < 1e-6)


def test_sample_normal():
    model = models.create_forward_model()
    model = normal(model, 'n', jnp.array([0.]), jnp.array([1.]))

    keys = jax.random.split(jax.random.PRNGKey(123), 500)
    samples = jax.vmap(lambda k, m: m.sample(k)['n'],
                       in_axes=(0, None),
                       out_axes=0)(keys, model)

    assert abs(jnp.mean(samples)) < 0.2

    model = models.create_forward_model()
    model = normal(model, 'n', jnp.array([10.]), jnp.array([1.]))

    keys = jax.random.split(jax.random.PRNGKey(123), 500)
    samples = jax.vmap(lambda k, m: m.sample(k)['n'],
                       in_axes=(0, None),
                       out_axes=0)(keys, model)

    assert abs(jnp.mean(samples)) - 10. < 0.2


def test_sample_normal_flexible_one_node():
    # test with one node
    model = models.create_forward_model()
    model = normal(model, 'n', jnp.zeros((10, 10), dtype=jnp.float32),
                   param.flexible_param(jnp.array(1.0)))

    key = jax.random.PRNGKey(123)
    sample = model.sample(key)['n']
    assert sample.shape == (10, 10)


def test_sample_normal_flexible_joint():
    # test with joint model
    model = models.create_forward_model()
    model = normal(model, 'weight', jnp.zeros((10, 10), dtype=jnp.float32),
                   jnp.zeros((10, 10), dtype=jnp.float32))
    model = normal(model, 'measurement', 'weight',
                   param.flexible_param(jnp.array(1.0)))

    key = jax.random.PRNGKey(123)
    sample = model.sample(key)['measurement']
    assert sample.shape == (10, 10)


def test_sample_joint_normal():
    model = models.create_forward_model()
    model = normal(model, 'weight', jnp.array([0.]), jnp.array([1.]))
    model = normal(model, 'measurement', 'weight', jnp.array([1.]))

    keys = jax.random.split(jax.random.PRNGKey(123), 500)
    samples = jax.vmap(lambda k, m: m.sample(k)['measurement'],
                       in_axes=(0, None),
                       out_axes=0)(keys, model)

    assert abs(jnp.mean(samples)) < 0.3


def test_incompatible_dimensions():
    model = models.create_forward_model()
    model = normal(model, 'weight', jnp.array([0., 0.]), jnp.array([1., 1.]))
    model = normal(model, 'measurement', 'weight', jnp.array([1.]))

    with pytest.raises(RuntimeError):
        key = jax.random.PRNGKey(123)
        model.sample(key)


def test_sample_conditioned():
    model = models.create_forward_model()
    model = normal(model, 'weight', jnp.array([0.]), jnp.array([1.]))
    model = normal(model, 'measurement', 'weight', jnp.array([1.]))
    model = func.condition(model, 'weight', jnp.array([0.]))

    keys = jax.random.split(jax.random.PRNGKey(123), 500)
    samples = jax.vmap(lambda k, m: m.sample(k)['measurement'],
                       in_axes=(0, None),
                       out_axes=0)(keys, model)

    assert abs(jnp.mean(samples)) < 0.2


def test_sample_conditioned_posterior_error():
    """This test should return an error as direct sampling from
    posterior is not possible.
    """
    model = models.create_forward_model()
    model = normal(model, 'n1', jnp.array([0.]), jnp.array([1.]))
    model = normal(model, 'n2', 'n1', jnp.array([1.]))
    model = func.condition(model, 'n2', jnp.array([0.]))

    key = jax.random.PRNGKey(123)
    with pytest.raises(RuntimeError):
        model.sample(key)

    model = models.create_forward_model()
    model = normal(model, 'n1', jnp.array([0.]), jnp.array([1.]))
    model = normal(model, 'n2', 'n1', jnp.array([1.]))
    model = normal(model, 'n3', 'n2', jnp.array([1.]))
    model = func.condition(model, 'n3', jnp.array([0.]))

    key = jax.random.PRNGKey(123)
    with pytest.raises(RuntimeError):
        model.sample(key)


def test_log_prob():
    model = models.create_forward_model()
    model = normal(model, 'n1', jnp.array([0.]), jnp.array([1.]))
    log_prob_mean = model.log_prob({'n1': jnp.array([0.])})
    assert(jnp.isclose(log_prob_mean, -0.9189385))
    log_prob_std = model.log_prob({'n1': jnp.array([1.])})
    assert(jnp.isclose(log_prob_std, -1.4189385))
