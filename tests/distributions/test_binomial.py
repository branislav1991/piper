# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

import pytest

import jax
import jax.numpy as jnp

import piper.functional as func
import piper.models as models
from piper.distributions import binomial
from piper.distributions import bernoulli
from piper import param


def test_sample_bernoulli():
    model = models.create_forward_model()
    model = bernoulli(model, 'n', jnp.array([0.5]))

    keys = jax.random.split(jax.random.PRNGKey(123), 100)
    samples = jax.vmap(lambda k, m: m.sample(k)['n'],
                       in_axes=(0, None),
                       out_axes=0)(keys, model)

    assert 0.4 < jnp.mean(samples) < 0.6

    model = models.create_forward_model()
    model = bernoulli(model, 'n', jnp.array([0.]))
    key = jax.random.PRNGKey(123)
    sample = model.sample(key)['n']
    assert sample == 0

    model = models.create_forward_model()
    model = bernoulli(model, 'n', jnp.array([1.]))
    key = jax.random.PRNGKey(123)
    sample = model.sample(key)['n']
    assert sample == 1

    model = models.create_forward_model()
    model = bernoulli(model, 'n', jnp.full((2, 2), 0.5))

    keys = jax.random.split(jax.random.PRNGKey(123), 100)
    samples = jax.vmap(lambda k, m: m.sample(k)['n'],
                       in_axes=(0, None),
                       out_axes=0)(keys, model)

    assert jnp.allclose(jnp.mean(samples, axis=0),
                        jnp.array([[0.45999998, 0.55], [0.57, 0.48]]))


def test_sample_binomial():
    model = models.create_forward_model()
    model = binomial(model, 'n', jnp.array([2]), jnp.array([0.5]))

    keys = jax.random.split(jax.random.PRNGKey(123), 100)
    samples = jax.vmap(lambda k, m: m.sample(k)['n'],
                       in_axes=(0, None),
                       out_axes=0)(keys, model)

    assert jnp.median(samples) == 1

    model = models.create_forward_model()
    model = binomial(model, 'n', jnp.array([1]), jnp.array([0.]))
    key = jax.random.PRNGKey(123)
    sample = model.sample(key)['n']
    assert sample == 0

    model = models.create_forward_model()
    model = binomial(model, 'n', jnp.array([10]), jnp.array([1.]))
    key = jax.random.PRNGKey(123)
    sample = model.sample(key)['n']
    assert sample == 10

    model = models.create_forward_model()
    model = binomial(model, 'n', jnp.array([10]), jnp.array([0.5]))

    keys = jax.random.split(jax.random.PRNGKey(123), 100)
    samples = jax.vmap(lambda k, m: m.sample(k)['n'],
                       in_axes=(0, None),
                       out_axes=0)(keys, model)

    assert jnp.median(samples) == 4


def test_sample_binomial_flexible_one_node():
    # test with one node
    model = models.create_forward_model()
    model = binomial(model, 'n', jnp.ones((10, 10), dtype=jnp.int32),
                     param.flexible_param(jnp.array(1.0)))

    key = jax.random.PRNGKey(123)
    sample = model.sample(key)['n']
    assert sample.shape == (10, 10) and jnp.all(sample == 1)


def test_kl_binomial_binomial_one_dimensional():
    model = models.create_forward_model()
    model = binomial(model, 'n1', jnp.array([10]), jnp.array([0.5]))
    model = binomial(model, 'n2', jnp.array([5]), jnp.array([0.5]))
    model = binomial(model, 'n3', jnp.array([10]), jnp.array([0.5]))
    model = binomial(model, 'n4', jnp.array([10]), jnp.array([0.1]))

    with pytest.raises(ValueError):
        func.compute_kl_div(model, 'n1', 'n2')

    assert func.compute_kl_div(model, 'n1', 'n3') < jnp.array(1e-6)
    assert func.compute_kl_div(model, 'n1', 'n4') - 5.1082563 < jnp.array(1e-6)


def test_kl_normal_normal_multi_dimensional():
    model = models.create_forward_model()
    model = binomial(model, 'n1', jnp.array([10, 5]), jnp.array([0.5, 0.5]))
    model = binomial(model, 'n2', jnp.array([5, 5]), jnp.array([0.5, 0.5]))
    model = binomial(model, 'n3', jnp.array([10, 5]), jnp.array([0.5, 0.5]))
    model = binomial(model, 'n4', jnp.array([10, 5]), jnp.array([0.1, 0.5]))

    with pytest.raises(ValueError):
        func.compute_kl_div(model, 'n1', 'n2')

    assert jnp.all(jnp.abs(func.compute_kl_div(model, 'n1', 'n3')) < 1e-6)
    assert jnp.all(
        jnp.abs(
            func.compute_kl_div(model, 'n1', 'n4')
            - jnp.array([5.1082563, 0.0])) < 1e-6)


def test_sample_conditioned():
    model = models.create_forward_model()
    model = bernoulli(model, 'n1', jnp.array([0.5]))
    model = binomial(model, 'n2', 'n1', jnp.array([0.5]))
    model = func.condition(model, 'n1', jnp.array([1]))

    keys = jax.random.split(jax.random.PRNGKey(123), 100)
    samples = jax.vmap(lambda k, m: m.sample(k)['n2'],
                       in_axes=(0, None),
                       out_axes=0)(keys, model)

    assert jnp.mean(samples) > 0.4 and jnp.mean(samples) < 0.6


def test_sample_conditioned_invalid_value_error():
    model = models.create_forward_model()
    model = bernoulli(model, 'n1', jnp.array([0.5]))
    model = binomial(model, 'n2', jnp.array([1]), 'n1')
    model = func.condition(model, 'n1', jnp.array([1]))

    key = jax.random.PRNGKey(123)
    with pytest.raises(TypeError):  # cannot condition p on int
        model.sample(key)


def test_sample_conditioned_posterior_error():
    """This test should return an error as direct sampling from
    posterior is not possible.
    """
    model = models.create_forward_model()
    model = bernoulli(model, 'n1', jnp.array([0.5]))
    model = binomial(model, 'n2', jnp.array([1]), 'n1')
    model = func.condition(model, 'n2', jnp.array([0]))

    key = jax.random.PRNGKey(123)
    with pytest.raises(RuntimeError):
        model.sample(key)


def test_log_prob_bernoulli():
    model = models.create_forward_model()
    model = bernoulli(model, 'n1', jnp.array([0.8]))
    log_prob_0 = model.log_prob({'n1': jnp.array([0])})
    log_prob_1 = model.log_prob({'n1': jnp.array([1])})
    assert (jnp.isclose(log_prob_0, -1.609438))
    assert (jnp.isclose(log_prob_1, -0.22314353))
