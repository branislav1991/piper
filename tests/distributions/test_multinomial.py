# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

import pytest

import jax
import jax.numpy as jnp

import piper.functional as func
from piper import distributions as dist
from piper import test_util as tu


def test_sample_categorical():
    def model1(key):
        return func.sample('n', dist.categorical(jnp.array([0.1, 0.8, 0.1])),
                           key)

    keys = jax.random.split(jax.random.PRNGKey(123), 100)
    samples = jax.vmap(lambda k: model1(k))(keys)

    assert jnp.argmax(jnp.sum(samples, 0)) == 1

    def model2(key):
        return func.sample(
            'n',
            dist.categorical(
                jnp.stack([
                    jnp.full((2, 2), 0.1),
                    jnp.full((2, 2), 0.8),
                    jnp.full((2, 2), 0.1)
                ], -1)), key)

    keys = jax.random.split(jax.random.PRNGKey(123), 100)
    samples = jax.vmap(lambda k: model2(k))(keys)

    assert jnp.all(jnp.argmax(jnp.sum(samples, 0), 2) == 1)


def test_sample_multinomial():
    def model1(key):
        return func.sample(
            'n', dist.multinomial(jnp.array(10), jnp.array([0.1, 0.8, 0.1])),
            key)

    keys = jax.random.split(jax.random.PRNGKey(123), 100)
    samples = jax.vmap(lambda k: model1(k))(keys)

    assert jnp.argmax(jnp.sum(samples, 0)) == 1

    def model2(key):
        return func.sample(
            'n',
            dist.multinomial(
                jnp.full((2, 2), 10),
                jnp.stack([
                    jnp.full((2, 2), 0.1),
                    jnp.full((2, 2), 0.8),
                    jnp.full((2, 2), 0.1)
                ], -1)), key)

    keys = jax.random.split(jax.random.PRNGKey(123), 100)
    samples = jax.vmap(lambda k: model2(k))(keys)

    assert jnp.all(jnp.argmax(jnp.sum(samples, 0), 2) == 1)


def test_kl_multinomial_multinomial_multi_dimensional():
    n1 = dist.multinomial(jnp.array([10, 5]),
                          jnp.array([[0.5, 0.5], [0.5, 0.5]]))
    n2 = dist.multinomial(jnp.array([5, 5]), jnp.array([[0.5, 0.5], [0.5,
                                                                     0.5]]))
    n3 = dist.multinomial(jnp.array([5, 5]),
                          jnp.array([[0.1, 0.8, 0.1], [0.1, 0.8, 0.1]]))
    n4 = dist.multinomial(jnp.array([10, 5]),
                          jnp.array([[0.1, 0.9], [0.1, 0.9]]))

    with pytest.raises(ValueError):
        func.compute_kl_div(n1, n2)

    with pytest.raises(ValueError):
        func.compute_kl_div(n1, n3)

    tu.check_close(func.compute_kl_div(n1, n4),
                   jnp.array([5.1082563, 2.5541282]))


def test_sample_multinomial_invalid_value_error():
    def model(key):
        keys = jax.random.split(key, 3)
        n1 = func.sample('n1', dist.categorical(jnp.array([0.5, 1.5])), keys[2])
        return n1

    key = jax.random.PRNGKey(123)
    sample = model(key)
    assert jnp.all(jnp.isnan(sample))


def test_sample_multinomial_invalid_condition():
    def model(key):
        n1 = func.sample('n1',
                         dist.categorical(jnp.array([0.1, 0.8, 0.1])),
                         key)
        return n1

    conditioned_model = func.condition(model, {'n1': jnp.array([2, 0, 0])})
    key = jax.random.PRNGKey(123)
    sample = conditioned_model(key)
    assert jnp.all(jnp.isnan(sample))  # more than 1 outcome


def test_log_prob_multinomial():
    n1 = dist.multinomial(jnp.array(10), jnp.array([0.1, 0.8, 0.1]))
    log_prob_first_10 = n1.log_prob(jnp.array([10, 0, 0]))
    log_prob_second_10 = n1.log_prob(jnp.array([0, 10, 0]))
    log_prob_third_10 = n1.log_prob(jnp.array([0, 0, 10]))

    log_prob_first_1_second_8_third_1 = n1.log_prob(jnp.array([1, 8, 1]))

    tu.check_close(log_prob_first_10, -23.025852)
    tu.check_close(log_prob_second_10, -2.2314363)
    tu.check_close(log_prob_third_10, -23.025852)
    tu.check_close(log_prob_first_1_second_8_third_1, -1.8905044)
