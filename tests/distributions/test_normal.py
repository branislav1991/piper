# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

import pytest

import jax
import jax.tree_util
import jax.numpy as jnp

import piper.functional as func
import piper.distributions as dist
from piper import test_util as tu


def test_kl_normal_normal_one_dimensional():
    n1 = dist.normal(jnp.array(0.), jnp.array(1.))
    n2 = dist.normal(jnp.array(0.), jnp.array(1.))
    n3 = dist.normal(jnp.array(1.), jnp.array(1.))
    n4 = dist.normal(jnp.array(1.), jnp.array(2.))

    assert func.compute_kl_div(n1, n2) < 1e-6
    assert func.compute_kl_div(n1, n3) - jnp.array(0.5) \
        < jnp.array(1e-6)
    assert func.compute_kl_div(n3, n4) \
        - jnp.array(0.09657359028) < jnp.array(1e-6)


def test_kl_normal_normal_multi_dimensional():
    n1 = dist.normal(jnp.array([0., 0.]), jnp.array([1., 1.]))
    n2 = dist.normal(jnp.array([0., 0.]), jnp.array([1., 1.]))
    n3 = dist.normal(jnp.array([1., 1.]), jnp.array([1., 1.]))
    n4 = dist.normal(jnp.array([1., 1.]), jnp.array([2., 2.]))

    assert jnp.all(func.compute_kl_div(n1, n2) < 1e-6)
    assert jnp.all(func.compute_kl_div(n1, n3) - 0.5 < 1e-6)
    assert jnp.all(func.compute_kl_div(n3, n4) - 0.09657359 < 1e-6)


def test_sample_normal():
    def model1(key):
        n = func.sample('n', dist.normal(jnp.array(0.), jnp.array(1.)), key)
        return n

    keys = jax.random.split(jax.random.PRNGKey(123), 100)
    samples = jax.vmap(lambda k: model1(k))(keys)

    assert abs(jnp.mean(samples)) < 0.2

    def model2(key):
        n = func.sample('n', dist.normal(jnp.array(10.), jnp.array(1.)), key)
        return n

    keys = jax.random.split(jax.random.PRNGKey(123), 100)
    samples = jax.vmap(lambda k: model2(k))(keys)

    assert abs(jnp.mean(samples)) - 10. < 0.2


def test_sample_joint_normal():
    def model(key):
        weight = func.sample('weight', dist.normal(jnp.array(0.),
                                                   jnp.array(1.)), key)
        measurement = func.sample('measurement',
                                  dist.normal(weight, jnp.array(1.)), key)
        return measurement

    keys = jax.random.split(jax.random.PRNGKey(123), 100)
    samples = jax.vmap(lambda k: model(k))(keys)
    assert abs(jnp.mean(samples)) < 0.3


def test_incompatible_dimensions():
    def model(key):
        weight = func.sample(
            'weight', dist.normal(jnp.array([0., 0.]), jnp.array([1., 1.])),
            key)
        measurement = func.sample('measurement',
                                  dist.normal(weight, jnp.array([1.])), key)
        return measurement

    with pytest.raises(ValueError):
        key = jax.random.PRNGKey(123)
        model(key)


def test_sample_conditioned():
    def model(key):
        weight = func.sample('weight', dist.normal(jnp.array(0.),
                                                   jnp.array(1.)), key)
        measurement = func.sample('measurement',
                                  dist.normal(weight, jnp.array(1.)), key)
        return measurement

    conditioned_model = func.condition(model, {'weight': jnp.array(0.)})
    keys = jax.random.split(jax.random.PRNGKey(123), 100)
    samples = jax.vmap(lambda k: conditioned_model(k))(keys)

    assert abs(jnp.mean(samples)) < 0.2


def test_log_prob_normal():
    n = dist.normal(jnp.array(0.), jnp.array(1.))
    log_prob_mean = n.log_prob(jnp.array(0.))
    tu.check_close(log_prob_mean, -0.9189385)
    log_prob_std = n.log_prob(jnp.array(1.))
    tu.check_close(log_prob_std, -1.4189385)
