# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

import pytest

import jax
import jax.numpy as jnp

import piper.functional as func
import piper.models as models
from piper.distributions import beta
from piper.distributions.binomial import bernoulli
from piper import param
from piper import test_util as tu


def test_sample_beta():
    model = models.create_forward_model()
    model = beta(model, 'n', jnp.array([0.5]), jnp.array([0.5]))

    keys = jax.random.split(jax.random.PRNGKey(123), 100)
    samples = jax.vmap(lambda k, m: m.sample(k)['n'],
                       in_axes=(0, None),
                       out_axes=0)(keys, model)

    tu.check_close(jnp.mean(samples), 0.45465213)

    model = models.create_forward_model()
    model = beta(model, 'n', jnp.array([2.]), jnp.array([5.]))
    samples = jax.vmap(lambda k, m: m.sample(k)['n'],
                       in_axes=(0, None),
                       out_axes=0)(keys, model)

    tu.check_close(jnp.mean(samples), 0.2994858)

    model = models.create_forward_model()
    model = beta(model, 'n', jnp.full((2, 2), 0.5), jnp.full((2, 2), 0.5))

    keys = jax.random.split(jax.random.PRNGKey(123), 100)
    samples = jax.vmap(lambda k, m: m.sample(k)['n'],
                       in_axes=(0, None),
                       out_axes=0)(keys, model)

    tu.check_close(
        jnp.mean(samples, 0),
        jnp.array([[0.51573753, 0.5368215], [0.50728726, 0.51721746]]))


def test_sample_beta_flexible_one_node():
    # test with one node
    model = models.create_forward_model()
    model = beta(model, 'n', jnp.array([0.5, 0.5]),
                 param.flexible_param(jnp.array(0.5)))

    key = jax.random.PRNGKey(123)
    sample = model.sample(key)['n']
    assert sample.shape == (2, )

    tu.check_close(sample, jnp.array([0.37926483, 0.9784184]))


def test_kl_beta_beta_one_dimensional():
    model = models.create_forward_model()
    model = beta(model, 'n1', jnp.array(1.0), jnp.array(0.5))
    model = beta(model, 'n2', jnp.array(2.0), jnp.array(0.5))

    tu.check_close(func.compute_kl_div(model, 'n1', 'n2'), 0.6137059)


def test_kl_beta_beta_multi_dimensional():
    model = models.create_forward_model()
    model = beta(model, 'n1', jnp.array([1.0, 2.0]), jnp.array([0.5, 0.5]))
    model = beta(model, 'n2', jnp.array([2.0, 1.0]), jnp.array([0.5, 0.5]))

    tu.check_close(func.compute_kl_div(model, 'n1', 'n2'),
                   jnp.array([0.6137059, -0.28037268]))


def test_sample_conditioned_invalid_value_error():
    model = models.create_forward_model()
    model = beta(model, 'n1', jnp.array(0.5), jnp.array(0.5))
    with pytest.raises(ValueError):  # value must be between 0 and 1
        model = func.condition(model, 'n1', jnp.array(2.0))


def test_sample_conditioned():
    model = models.create_forward_model()
    model = beta(model, 'n1', jnp.array([0.5]), jnp.array([0.5]))
    model = bernoulli(model, 'n2', 'n1')
    model = func.condition(model, 'n1', jnp.array([1.0]))

    keys = jax.random.split(jax.random.PRNGKey(123), 100)
    samples = jax.vmap(lambda k, m: m.sample(k)['n2'],
                       in_axes=(0, None),
                       out_axes=0)(keys, model)

    assert jnp.all(samples == 1)


def test_log_prob_beta():
    model = models.create_forward_model()
    model = beta(model, 'n1', jnp.array(0.5), jnp.array(0.5))
    log_prob_1 = model.log_prob({'n1': jnp.array(0.1)})
    log_prob_2 = model.log_prob({'n1': jnp.array(0.9)})
    tu.check_close(log_prob_1, 0.05924129)
    tu.check_close(log_prob_2, 0.05924118)
