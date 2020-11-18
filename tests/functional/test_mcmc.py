# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

import jax
import jax.numpy as jnp

import piper
import piper.functional as func
import piper.distributions as dist
import piper.models as models


def test_mcmc():
    m = models.create_forward_model()
    m = dist.normal(m, 'n1', jnp.array([0.]), jnp.array([1.]))
    m = dist.normal(m, 'n2', 'n1', jnp.array([1.]))

    m = func.condition(m, 'n2', jnp.array([0.5]))

    proposal = models.create_forward_model()
    proposal = dist.normal(proposal, 'n1', jnp.array([0.]), jnp.array([1.]))
    initial_params = {'n1': 0.}
    mcmc_model = func.mcmc(m, proposal, initial_params, burnin_steps=500)
    mcmc_model = func.burnin(mcmc_model)

    samples = []
    keys = jax.random.split(jax.random.PRNGKey(123), 100)
    for i in range(100):
        samples.append(piper.sample(mcmc_model, keys[i]))
