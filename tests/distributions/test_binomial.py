# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

import pytest

import jax
import jax.numpy as jnp

import piper
import piper.functional as func
from piper.distributions import binomial
from piper.distributions import bernoulli
from piper import param


def test_throw_bad_params():
    # Should fail because of missing model
    with pytest.raises(ValueError):
        binomial(None, 'test', jnp.array([1]), jnp.array([0.5]))

    # Should fail because of wrong parameters
    model = piper.create_graph()
    with pytest.raises(TypeError):
        binomial(model, 'test2', 1, 0.5)

    with pytest.raises(ValueError):
        binomial(model, 'test2', jnp.array([-1]), jnp.array([0.5]))

    with pytest.raises(ValueError):
        binomial(model, 'test2', jnp.array([1]), jnp.array([1.5]))

    # Should fail because of different shapes of n and p
    with pytest.raises(ValueError):
        binomial(model, 'test3', jnp.array([1, 1]), jnp.array([0.5]))


def test_sample_bernoulli():
    model = piper.create_graph()
    model = bernoulli(model, 'n', jnp.array([0.5]))

    keys = jax.random.split(jax.random.PRNGKey(123), 500)
    samples = jax.vmap(lambda k, m: func.sample(m, k)['n'].value,
                       in_axes=(0, None),
                       out_axes=0)(keys, model)

    assert 0.4 < jnp.mean(samples) < 0.6

    model = piper.create_graph()
    model = bernoulli(model, 'n', jnp.array([0.]))
    key = jax.random.PRNGKey(123)
    sample = func.sample(model, key)['n'].value
    assert sample == 0

    model = piper.create_graph()
    model = bernoulli(model, 'n', jnp.array([1.]))
    key = jax.random.PRNGKey(123)
    sample = func.sample(model, key)['n'].value
    assert sample == 1

    model = piper.create_graph()
    model = bernoulli(model, 'n', jnp.full((10, 10), 0.5))

    keys = jax.random.split(jax.random.PRNGKey(123), 500)
    samples = jax.vmap(lambda k, m: func.sample(m, k)['n'].value,
                       in_axes=(0, None),
                       out_axes=0)(keys, model)

    assert jnp.all(0.4 < jnp.mean(samples, axis=0)) and jnp.all(
        jnp.mean(samples, axis=0) < 0.6)


def test_sample_binomial():
    model = piper.create_graph()
    model = binomial(model, 'n', jnp.array([2]), jnp.array([0.5]))

    keys = jax.random.split(jax.random.PRNGKey(123), 500)
    samples = jax.vmap(lambda k, m: func.sample(m, k)['n'].value,
                       in_axes=(0, None),
                       out_axes=0)(keys, model)

    assert jnp.median(samples) == 1

    model = piper.create_graph()
    model = binomial(model, 'n', jnp.array([1]), jnp.array([0.]))
    key = jax.random.PRNGKey(123)
    sample = func.sample(model, key)['n'].value
    assert sample == 0

    model = piper.create_graph()
    model = binomial(model, 'n', jnp.array([10]), jnp.array([1.]))
    key = jax.random.PRNGKey(123)
    sample = func.sample(model, key)['n'].value
    assert sample == 10

    model = piper.create_graph()
    model = binomial(model, 'n', jnp.array([10]), jnp.array([0.5]))

    keys = jax.random.split(jax.random.PRNGKey(123), 500)
    samples = jax.vmap(lambda k, m: func.sample(m, k)['n'].value,
                       in_axes=(0, None),
                       out_axes=0)(keys, model)

    assert jnp.median(samples) == 5


def test_sample_binomial_all_flexible_error():
    # all params are flexible - not possible
    with pytest.raises(ValueError):
        model = piper.create_graph()
        model = binomial(model, 'n', param.flexible_param(jnp.array(1)),
                         param.flexible_param(jnp.array(0.5)))


def test_sample_binomial_flexible_one_node():
    # test with one node
    model = piper.create_graph()
    model = binomial(model, 'n', jnp.ones((10, 10), dtype=jnp.float32),
                     param.flexible_param(jnp.array(1.0)))

    key = jax.random.PRNGKey(123)
    sample = func.sample(model, key)['n'].value
    assert sample.shape == (10, 10) and jnp.all(sample == 1)


# def test_sample_binomial_flexible_joint():
#     # test with joint model
#     model = piper.create_graph()
#     model = binomial(model, 'weight', jnp.zeros((10, 10), dtype=jnp.float32),
#                    jnp.zeros((10, 10), dtype=jnp.float32))
#     model = binomial(model, 'measurement', 'weight',
#                    param.flexible_param(jnp.array(1.0)))

#     key = jax.random.PRNGKey(123)
#     sample = func.sample(model, key)['measurement'].value
#     assert sample.shape == (10, 10)


def test_kl_binomial_binomial_one_dimensional():
    model = piper.create_graph()
    model = binomial(model, 'n1', jnp.array([10]), jnp.array([0.5]))
    model = binomial(model, 'n2', jnp.array([5]), jnp.array([0.5]))
    model = binomial(model, 'n3', jnp.array([10]), jnp.array([0.5]))
    model = binomial(model, 'n4', jnp.array([10]), jnp.array([0.0]))
    model = binomial(model, 'n5', jnp.array([10]), jnp.array([0.1]))

    with pytest.raises(ValueError):
        func.kl_divergence(model, 'n1', 'n2')

    with pytest.warns(UserWarning):  # will be inf
        func.kl_divergence(model, 'n1', 'n4')

    with pytest.warns(UserWarning):  # will be nan
        func.kl_divergence(model, 'n4', 'n3')

    assert func.kl_divergence(model, 'n1', 'n3') < jnp.array(1e-6)
    assert func.kl_divergence(model, 'n1', 'n5') - 5.1082563 < jnp.array(1e-6)


def test_kl_normal_normal_multi_dimensional():
    model = piper.create_graph()
    model = binomial(model, 'n1', jnp.array([10, 5]), jnp.array([0.5, 0.5]))
    model = binomial(model, 'n2', jnp.array([5, 5]), jnp.array([0.5, 0.5]))
    model = binomial(model, 'n3', jnp.array([10, 5]), jnp.array([0.5, 0.5]))
    model = binomial(model, 'n4', jnp.array([10, 5]), jnp.array([0.0, 0.5]))
    model = binomial(model, 'n5', jnp.array([10, 5]), jnp.array([0.1, 0.5]))

    with pytest.raises(ValueError):
        func.kl_divergence(model, 'n1', 'n2')

    with pytest.warns(UserWarning):  # will be inf
        func.kl_divergence(model, 'n1', 'n4')

    with pytest.warns(UserWarning):  # will be nan
        func.kl_divergence(model, 'n4', 'n3')

    assert jnp.all(jnp.abs(func.kl_divergence(model, 'n1', 'n3')) < 1e-6)
    assert jnp.all(
        jnp.abs(
            func.kl_divergence(model, 'n1', 'n5')
            - jnp.array([5.1082563, 0.0])) < 1e-6)


# def test_sample_beta_binomial():
#     model = normal(model, 'weight', jnp.array([0.]), jnp.array([1.]))
#     model = normal(model, 'measurement', 'weight', jnp.array([1.]))
#     samples = []
#     for i in range(500):
#         samples.append(func.sample(model)['measurement'])

#     samples = jnp.stack(samples)
#     assert abs(jnp.mean(samples)) < 0.1

# def test_incompatible_dimensions():
#     model = piper.create_graph()
#     model = binomial(model, 'weight', jnp.array([0., 0.]),
#                      jnp.array([1., 1.]))
#     model = normal(model, 'measurement', 'weight', jnp.array([1.]))

#     with pytest.raises(RuntimeError):
#         func.sample(model)
