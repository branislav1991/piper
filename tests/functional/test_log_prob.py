# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

import jax
import jax.numpy as jnp

import piper.functional as func
import piper.distributions as dist
from piper import test_util as tu


def test_log_prob_normal():
    def model(key):
        n1 = func.sample('n1', dist.normal(jnp.array(0.), jnp.array(1.)), key)
        return n1

    # Unconditioned
    tracer = func.trace(model)
    key = jax.random.PRNGKey(123)
    tracer(key)
    tree = tracer.get_tree()
    log_prob = func.log_prob(tree)

    tu.check_close(log_prob, -1.2025023)

    # Conditioned
    conditioned_model = func.condition(model, {'n1': jnp.array(0.)})
    tracer = func.trace(conditioned_model)
    key = jax.random.PRNGKey(123)
    tracer(key)
    tree = tracer.get_tree()
    log_prob = func.log_prob(tree)

    tu.check_close(log_prob, -0.9189385)


def test_log_prob_normal_normal():
    def model(key):
        keys = jax.random.split(key)
        n1 = func.sample('n1',
                         dist.normal(jnp.array(0.), jnp.array(1.)),
                         keys[0])
        n2 = func.sample('n2', dist.normal(n1, jnp.array(1.)), keys[1])
        return n2

    conditioned_model = func.condition(model, {
        'n1': jnp.array(0.),
        'n2': jnp.array(1.)
    })
    tracer = func.trace(conditioned_model)
    key = jax.random.PRNGKey(123)
    tracer(key)
    tree = tracer.get_tree()
    log_prob = func.log_prob(tree)

    tu.check_close(log_prob, -2.337877)
