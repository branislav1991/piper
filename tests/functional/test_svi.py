# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

import jax
import jax.numpy as jnp
import jax.experimental.optimizers as jax_optim

import piper.functional as func
import piper.distributions as dist
from piper import test_util as tu


def test_svi():
    def model(key):
        n1 = func.sample('n1', dist.normal(jnp.array(10.), jnp.array(10.)),
                         key)
        return n1

    def q(params, key):
        n1 = func.sample('n1', dist.normal(params['n1_mean'],
                                           params['n1_std']), key)
        return {'n1': n1}

    optimizer = jax_optim.adam(0.05)
    svi = func.svi(model,
                   q,
                   func.elbo,
                   optimizer,
                   initial_params={
                       'n1_mean': jnp.array(0.),
                       'n1_std': jnp.array(1.)
                   })

    keys = jax.random.split(jax.random.PRNGKey(123), 500)
    for i in range(500):
        loss = svi(keys[i])
        if i % 100 == 0:
            print(f"Step {i}: {loss}")

    inferred_n1_mean = svi.get_param('n1_mean')
    inferred_n1_std = svi.get_param('n1_std')
    tu.check_close(inferred_n1_mean, 10.611818)
    tu.check_close(inferred_n1_std, 9.024648)
