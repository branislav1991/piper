# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

import jax
import jax.tree_util
import jax.numpy as jnp

import piper.functional as func
import piper.distributions as dist
from piper import test_util as tu


def test_kl_uniform_uniform():
    n1 = dist.uniform()
    n2 = dist.uniform()

    assert func.compute_kl_div(n1, n2) == 0


def test_sample_uniform():
    def model1(key):
        n = func.sample('n', dist.uniform(), key)
        return n

    keys = jax.random.split(jax.random.PRNGKey(123), 100)
    samples = jax.vmap(lambda k: model1(k))(keys)

    assert abs(jnp.mean(samples)) - 0.5 < 0.1

    def model2(key):
        n = func.sample('n', dist.uniform((2, 2)), key)
        return n

    keys = jax.random.split(jax.random.PRNGKey(123), 100)
    samples = jax.vmap(lambda k: model2(k))(keys)

    assert jnp.all(abs(jnp.mean(samples, 0)) - 0.5 < 0.1)


def test_sample_joint_normal():
    def model(key):
        keys = jax.random.split(key)
        weight = func.sample('weight',
                             dist.normal(jnp.array(0.), jnp.array(1.)),
                             keys[0])
        measurement = func.sample('measurement',
                                  dist.normal(weight, jnp.array(1.)),
                                  keys[1])
        return measurement

    keys = jax.random.split(jax.random.PRNGKey(123), 100)
    samples = jax.vmap(lambda k: model(k))(keys)
    assert abs(jnp.mean(samples)) < 0.3


def test_log_prob_uniform():
    n = dist.uniform()
    log_prob_0 = n.log_prob(jnp.array(0.))
    log_prob_1 = n.log_prob(jnp.array(1.))
    tu.check_close(log_prob_0, 0)
    tu.check_close(log_prob_1, 0)
