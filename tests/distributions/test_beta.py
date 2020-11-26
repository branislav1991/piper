# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

import pytest

import jax
import jax.numpy as jnp

import piper.functional as func
from piper import distributions as dist
from piper import test_util as tu


def test_sample_beta():
    def model1(key):
        return func.sample('n', dist.beta(jnp.array(0.5), jnp.array(0.5)), key)

    keys = jax.random.split(jax.random.PRNGKey(123), 100)
    samples = jax.vmap(lambda k: model1(k))(keys)

    tu.check_close(jnp.mean(samples), 0.45465213)

    def model2(key):
        return func.sample('n', dist.beta(jnp.array(2.), jnp.array(5.)), key)

    samples = jax.vmap(lambda k: model2(k))(keys)

    tu.check_close(jnp.mean(samples), 0.2994858)

    def model3(key):
        return func.sample('n', dist.beta(jnp.full((2, 2), 0.5), jnp.full((2, 2), 0.5)), key)

    keys = jax.random.split(jax.random.PRNGKey(123), 100)
    samples = jax.vmap(lambda k: model3(k))(keys)

    tu.check_close(
        jnp.mean(samples, 0),
        jnp.array([[0.51573753, 0.5368215], [0.50728726, 0.51721746]]))


def test_kl_beta_beta_one_dimensional():
    n1 = dist.beta(jnp.array(1.0), jnp.array(0.5))
    n2 = dist.beta(jnp.array(2.0), jnp.array(0.5))

    tu.check_close(func.compute_kl_div(n1, n2), 0.6137059)


def test_kl_beta_beta_multi_dimensional():
    n1 = dist.beta(jnp.array([1.0, 2.0]), jnp.array([0.5, 0.5]))
    n2 = dist.beta(jnp.array([2.0, 1.0]), jnp.array([0.5, 0.5]))

    tu.check_close(func.compute_kl_div(n1, n2),
                   jnp.array([0.6137059, -0.28037268]))


def test_sample_conditioned_invalid_value_error():
    def model(key):
        n1 = func.sample('n1', dist.beta(jnp.array(0.5), jnp.array(0.5)), key)
        return n1

    with pytest.raises(ValueError):  # value must be between 0 and 1
        with func.Condition({'n1': jnp.array(2.0)}):
            key = jax.random.PRNGKey(123)
            model(key)


def test_sample_conditioned():
    def model(key):
        n1 = func.sample('n1', dist.beta(jnp.array(0.5), jnp.array(0.5)), key)
        n2 = func.sample('n2', dist.bernoulli(n1), key)
        return n2

    with func.Condition({'n1': jnp.array(1.0)}):
        keys = jax.random.split(jax.random.PRNGKey(123), 100)
        samples = jax.vmap(lambda k: model(k))(keys)

    assert jnp.all(samples == 1)


def test_log_prob_beta():
    n1 = dist.beta(jnp.array(0.5), jnp.array(0.5))
    log_prob_1 = n1.log_prob(jnp.array(0.1))
    log_prob_2 = n1.log_prob(jnp.array(0.9))
    tu.check_close(log_prob_1, 0.05924129)
    tu.check_close(log_prob_2, 0.05924118)
