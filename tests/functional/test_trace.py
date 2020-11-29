# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

import jax
import jax.numpy as jnp

import piper.functional as func
import piper.distributions as dist


def test_trace_normal_normal():
    def model(key):
        keys = jax.random.split(key)
        n1 = func.sample('n1', dist.normal(jnp.array(0.), jnp.array(1.)),
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

    assert 'n1' in tree.nodes \
        and tree.nodes['n1'].distribution.mu == 0. \
        and tree.nodes['n1'].distribution.sigma == 1. \
        and tree.nodes['n1'].value == 0.

    assert 'n2' in tree.nodes \
        and tree.nodes['n2'].distribution.mu == 0. \
        and tree.nodes['n2'].distribution.sigma == 1. \
        and tree.nodes['n2'].value == 1.
