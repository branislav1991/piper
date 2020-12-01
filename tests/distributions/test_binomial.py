# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

import pytest

import jax
import jax.numpy as jnp

import piper.functional as func
from piper import distributions as dist
from piper import test_util as tu


def test_sample_bernoulli():
    def model1(key):
        return func.sample('n', dist.bernoulli(jnp.array(0.5)), key)

    keys = jax.random.split(jax.random.PRNGKey(123), 100)
    samples = jax.vmap(lambda k: model1(k))(keys)

    assert 0.4 < jnp.mean(samples) < 0.6

    def model2(key):
        return func.sample('n', dist.bernoulli(jnp.array(0.)), key)

    key = jax.random.PRNGKey(123)
    sample = model2(key)
    assert sample == 0

    def model3(key):
        return func.sample('n', dist.bernoulli(jnp.array(1.)), key)

    key = jax.random.PRNGKey(123)
    sample = model3(key)
    assert sample == 1

    def model4(key):
        return func.sample('n', dist.bernoulli(jnp.full((2, 2), 0.5)), key)

    keys = jax.random.split(jax.random.PRNGKey(123), 100)
    samples = jax.vmap(lambda k: model4(k))(keys)

    tu.check_close(jnp.mean(samples, axis=0),
                   jnp.array([[0.45999998, 0.55], [0.57, 0.48]]))


def test_sample_binomial():
    def model1(key):
        return func.sample('n', dist.binomial(jnp.array(2), jnp.array(0.5)),
                           key)

    keys = jax.random.split(jax.random.PRNGKey(123), 100)
    samples = jax.vmap(lambda k: model1(k))(keys)

    assert jnp.median(samples) == 1

    def model2(key):
        return func.sample('n', dist.binomial(jnp.array(1), jnp.array(0.)),
                           key)

    key = jax.random.PRNGKey(123)
    sample = model2(key)
    assert sample == 0

    def model3(key):
        return func.sample('n', dist.binomial(jnp.array(10), jnp.array(1.)),
                           key)

    key = jax.random.PRNGKey(123)
    sample = model3(key)
    assert sample == 10

    def model4(key):
        return func.sample('n', dist.binomial(jnp.array(10), jnp.array(0.5)),
                           key)

    keys = jax.random.split(jax.random.PRNGKey(123), 100)
    samples = jax.vmap(lambda k: model4(k))(keys)

    assert jnp.median(samples) == 4


def test_kl_binomial_binomial_one_dimensional():
    n1 = dist.binomial(jnp.array(10), jnp.array(0.5))
    n2 = dist.binomial(jnp.array(5), jnp.array(0.5))
    n3 = dist.binomial(jnp.array(10), jnp.array(0.5))
    n4 = dist.binomial(jnp.array(10), jnp.array(0.1))

    with pytest.raises(ValueError):
        func.compute_kl_div(n1, n2)

    assert func.compute_kl_div(n1, n3) < jnp.array(1e-6)
    assert func.compute_kl_div(n1, n4) - 5.1082563 < jnp.array(1e-6)


def test_kl_normal_normal_multi_dimensional():
    n1 = dist.binomial(jnp.array([10, 5]), jnp.array([0.5, 0.5]))
    n2 = dist.binomial(jnp.array([5, 5]), jnp.array([0.5, 0.5]))
    n3 = dist.binomial(jnp.array([10, 5]), jnp.array([0.5, 0.5]))
    n4 = dist.binomial(jnp.array([10, 5]), jnp.array([0.1, 0.5]))

    with pytest.raises(ValueError):
        func.compute_kl_div(n1, n2)

    assert jnp.all(jnp.abs(func.compute_kl_div(n1, n3)) < 1e-6)
    assert jnp.all(
        jnp.abs(func.compute_kl_div(n1, n4)
                - jnp.array([5.1082563, 0.0])) < 1e-6)


def test_sample_conditioned():
    def model(key):
        keys = jax.random.split(key)
        n1 = func.sample('n1', dist.bernoulli(jnp.array(0.5)), keys[0])
        n2 = func.sample('n2', dist.binomial(n1, jnp.array(0.5)), keys[1])
        return n2

    conditioned_model = func.condition(model, {'n1': jnp.array(1)})
    keys = jax.random.split(jax.random.PRNGKey(123), 100)
    samples = jax.vmap(lambda k: conditioned_model(k))(keys)

    assert jnp.mean(samples) > 0.4 and jnp.mean(samples) < 0.6


def test_sample_binomial_invalid_value_error():
    def model(key):
        keys = jax.random.split(key)
        n1 = func.sample('n1', dist.bernoulli(jnp.array(2.0)), keys[1])
        return n1

    key = jax.random.PRNGKey(123)
    sample = model(key)
    assert jnp.isnan(sample)


def test_sample_binomial_invalid_condition():
    def model(key):
        keys = jax.random.split(key)
        n1 = func.sample('n1', dist.bernoulli(jnp.array(0.5)), keys[0])
        n2 = func.sample('n2', dist.binomial(jnp.array(1), n1), keys[1])
        return n2

    conditioned_model = func.condition(model, {'n1': jnp.array(0.5)})
    key = jax.random.PRNGKey(123)
    sample = conditioned_model(key)
    assert jnp.isnan(sample)

    conditioned_model = func.condition(model, {'n1': jnp.array(2)})
    key = jax.random.PRNGKey(123)
    sample = conditioned_model(key)
    assert jnp.isnan(sample)


def test_log_prob_bernoulli():
    n1 = dist.bernoulli(jnp.array(0.8))
    log_prob_0 = n1.log_prob(0)
    log_prob_1 = n1.log_prob(1)
    tu.check_close(log_prob_0, -1.609438)
    tu.check_close(log_prob_1, -0.22314353)
