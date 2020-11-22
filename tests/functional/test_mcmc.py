# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

import pytest

import jax
import jax.numpy as jnp

import piper.functional as func
import piper.distributions as dist
import piper.models as models


def test_mcmc_wrong_proposal_or_initial_samples():
    m = models.create_forward_model()
    m = dist.normal(m, 'n1', jnp.array([0.]), jnp.array([1.]))
    m = dist.normal(m, 'n2', 'n1', jnp.array([1.]))

    m = func.condition(m, 'n2', jnp.array([0.5]))

    proposal = models.create_forward_model()
    proposal = dist.normal(proposal, 'n3', jnp.array([0.]), jnp.array([1.]))
    initial_samples = {'n1': 0.}
    with pytest.raises(KeyError):
        func.mcmc(m, proposal, initial_samples, burnin_steps=500,
                  num_chains=1)  # incorrect proposal

    proposal = models.create_forward_model()
    proposal = dist.normal(proposal, 'n1', jnp.array([0.]), jnp.array([1.]))
    initial_samples = {'n3': 0.}
    with pytest.raises(KeyError):
        func.mcmc(m, proposal, initial_samples, burnin_steps=500,
                  num_chains=1)  # incorrect initial samples


def test_mcmc_normal_normal_one_chain():
    m = models.create_forward_model()
    m = dist.normal(m, 'n1', jnp.array([0.]), jnp.array([1.]))
    m = dist.normal(m, 'n2', 'n1', jnp.array([1.]))

    m = func.condition(m, 'n2', jnp.array([0.5]))

    proposal = models.create_forward_model()
    proposal = dist.normal(proposal, 'n1', jnp.array([0.]), jnp.array([10.]))
    initial_samples = {'n1': jnp.array([0.])}
    mcmc_model = func.mcmc(m,
                           proposal,
                           initial_samples,
                           burnin_steps=100,
                           num_chains=1)

    samples = []
    keys = jax.random.split(jax.random.PRNGKey(123), 100)
    for i in range(100):
        samples.append(mcmc_model.sample(keys[i]))

    n1_samples = jnp.stack([i['n1'] for i in samples])
    assert jnp.isclose(jnp.mean(n1_samples), 0.29319033)


def test_mcmc_normal_normal_10_chains():
    m = models.create_forward_model()
    m = dist.normal(m, 'n1', jnp.array([0.]), jnp.array([1.]))
    m = dist.normal(m, 'n2', 'n1', jnp.array([1.]))

    m = func.condition(m, 'n2', jnp.array([0.5]))

    proposal = models.create_forward_model()
    proposal = dist.normal(proposal, 'n1', jnp.zeros((10, )),
                           jnp.full((10, ), 10.))

    initial_samples = {'n1': jnp.zeros((10, ))}
    mcmc_model = func.mcmc(m,
                           proposal,
                           initial_samples,
                           burnin_steps=100,
                           num_chains=10)

    samples = []
    keys = jax.random.split(jax.random.PRNGKey(123), 100)
    for i in range(100):
        samples.append(mcmc_model.sample(keys[i]))

    n1_samples = jnp.stack([i['n1'] for i in samples]).reshape((-1, ))
    assert jnp.isclose(jnp.mean(n1_samples), 0.26838708)


def test_mcmc_normal_normal_multidim_10_chains():
    key = jax.random.PRNGKey(123)
    key, sk = jax.random.split(key)
    m = models.create_forward_model()
    m = dist.normal(m, 'n1', jax.random.normal(sk, (2, 2)), jnp.ones((2, 2)))
    m = dist.normal(m, 'n2', 'n1', jnp.ones((2, 2)))

    m = func.condition(m, 'n2', jnp.full((2, 2), 0.5))

    proposal = models.create_forward_model()
    proposal = dist.normal(proposal, 'n1', jnp.zeros((3, 2, 2)),
                           jnp.full((3, 2, 2), 10.))
    initial_samples = {'n1': jnp.zeros((3, 2, 2))}
    mcmc_model = func.mcmc(m,
                           proposal,
                           initial_samples,
                           burnin_steps=100,
                           num_chains=3)

    samples = []
    keys = jax.random.split(jax.random.PRNGKey(123), 100)
    for i in range(100):
        samples.append(mcmc_model.sample(keys[i]))

    n1_samples = jnp.stack([i['n1'] for i in samples]).reshape((-1, 2, 2))
    assert jnp.allclose(jnp.mean(n1_samples, 0), jnp.array([[0.12954064, 0.06247671], [0.47244635, 0.4246871]]))
