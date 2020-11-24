# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

import pytest

import jax
import jax.numpy as jnp

import piper.core as core
import piper.functional as func
import piper.distributions as dist
import piper.models as models


def test_metropolis_hastings_wrong_proposal_or_initial_samples():
    m = models.create_forward_model()
    m = dist.normal(m, 'n1', jnp.array([0.]), jnp.array([1.]))
    m = dist.normal(m, 'n2', 'n1', jnp.array([1.]))

    m = func.condition(m, 'n2', jnp.array([0.5]))

    proposal = models.create_forward_model()
    proposal = core.const_node(proposal, 'n3', jnp.array([0.]))
    proposal = dist.normal(proposal, 'n4', 'n3', jnp.array([1.]))
    proposal = func.proposal(proposal, {'n3': 'n4'})
    initial_samples = {'n1': 0.}
    with pytest.raises(KeyError):
        func.metropolis_hastings(m,
                                 proposal,
                                 initial_samples,
                                 burnin_steps=100,
                                 num_chains=1)  # incorrect proposal

    proposal = models.create_forward_model()
    proposal = core.const_node(proposal, 'n4', jnp.array([0.]))
    proposal = dist.normal(proposal, 'n1', 'n4', jnp.array([1.]))
    proposal = func.proposal(proposal, {'n4': 'n1'})
    initial_samples = {'n3': 0.}
    with pytest.raises(KeyError):
        func.metropolis_hastings(m,
                                 proposal,
                                 initial_samples,
                                 burnin_steps=100,
                                 num_chains=1)  # incorrect initial samples


def test_metropolis_hastings_normal_one_chain():
    m = models.create_forward_model()
    m = dist.normal(m, 'n1', jnp.array([0.]), jnp.array([1.]))

    proposal = models.create_forward_model()
    proposal = core.const_node(proposal, 'n2', jnp.array([0.]))
    proposal = dist.normal(proposal, 'n1', 'n2', jnp.array([5.0]))
    proposal = func.proposal(proposal, {'n2': 'n1'})
    initial_samples = {'n1': jnp.array([0.])}
    metropolis_hastings_model = func.metropolis_hastings(m,
                                                         proposal,
                                                         initial_samples,
                                                         burnin_steps=300,
                                                         num_chains=1)

    samples = []
    keys = jax.random.split(jax.random.PRNGKey(123), 100)
    for i in range(100):
        samples.append(metropolis_hastings_model.sample(keys[i]))

    n1_samples = jnp.stack([i['n1'] for i in samples])
    assert jnp.isclose(jnp.mean(n1_samples), -0.08231668)


def test_metropolis_hastings_normal_normal_one_chain():
    m = models.create_forward_model()
    m = dist.normal(m, 'n1', jnp.array([0.]), jnp.array([1.]))
    m = dist.normal(m, 'n2', 'n1', jnp.array([1.]))

    m = func.condition(m, 'n2', jnp.array([0.5]))

    proposal = models.create_forward_model()
    proposal = core.const_node(proposal, 'n3', jnp.array([0.]))
    proposal = dist.normal(proposal, 'n1', 'n3', jnp.array([5.0]))
    proposal = func.proposal(proposal, {'n3': 'n1'})
    initial_samples = {'n1': jnp.array([0.])}
    metropolis_hastings_model = func.metropolis_hastings(m,
                                                         proposal,
                                                         initial_samples,
                                                         burnin_steps=300,
                                                         num_chains=1)

    samples = []
    keys = jax.random.split(jax.random.PRNGKey(123), 100)
    for i in range(100):
        samples.append(metropolis_hastings_model.sample(keys[i]))

    n1_samples = jnp.stack([i['n1'] for i in samples])
    assert jnp.isclose(jnp.mean(n1_samples), 0.24618316)


def test_metropolis_hastings_normal_normal_10_chains():
    m = models.create_forward_model()
    m = dist.normal(m, 'n1', jnp.array([0.]), jnp.array([1.]))
    m = dist.normal(m, 'n2', 'n1', jnp.array([1.]))

    m = func.condition(m, 'n2', jnp.array([0.5]))

    proposal = models.create_forward_model()
    proposal = core.const_node(proposal, 'n3', jnp.zeros((10, )))
    proposal = dist.normal(proposal, 'n1', 'n3', jnp.full((10, ), 5.0))
    proposal = func.proposal(proposal, {'n3': 'n1'})

    initial_samples = {'n1': jnp.zeros((10, ))}
    metropolis_hastings_model = func.metropolis_hastings(m,
                                                         proposal,
                                                         initial_samples,
                                                         burnin_steps=300,
                                                         num_chains=10)

    samples = []
    keys = jax.random.split(jax.random.PRNGKey(123), 100)
    for i in range(100):
        samples.append(metropolis_hastings_model.sample(keys[i]))

    n1_samples = jnp.stack([i['n1'] for i in samples]).reshape((-1, ))
    assert jnp.isclose(jnp.mean(n1_samples), 0.15941639)


def test_metropolis_hastings_normal_normal_multidim_10_chains():
    key = jax.random.PRNGKey(123)
    key, sk = jax.random.split(key)
    m = models.create_forward_model()
    m = dist.normal(m, 'n1', jax.random.normal(sk, (2, 2)), jnp.ones((2, 2)))
    m = dist.normal(m, 'n2', 'n1', jnp.ones((2, 2)))

    m = func.condition(m, 'n2', jnp.full((2, 2), 0.5))

    proposal = models.create_forward_model()
    proposal = core.const_node(proposal, 'n3', jnp.zeros((3, 2, 2)))
    proposal = dist.normal(proposal, 'n1', 'n3', jnp.full((3, 2, 2), 5.0))
    proposal = func.proposal(proposal, {'n3': 'n1'})
    initial_samples = {'n1': jnp.zeros((3, 2, 2))}
    metropolis_hastings_model = func.metropolis_hastings(m,
                                                         proposal,
                                                         initial_samples,
                                                         burnin_steps=300,
                                                         num_chains=3)

    samples = []
    keys = jax.random.split(jax.random.PRNGKey(123), 100)
    for i in range(100):
        samples.append(metropolis_hastings_model.sample(keys[i]))

    n1_samples = jnp.stack([i['n1'] for i in samples]).reshape((-1, 2, 2))
    assert jnp.allclose(
        jnp.mean(n1_samples, 0),
        jnp.array([[-0.22292876, 0.19647372], [0.39852002, 0.36776695]]))


# def test_metropolis_hastings_beta_bernoulli_10chains():
    # m = models.create_forward_model()
    # m = dist.beta(m, 'n1', jnp.array([0.5]), jnp.array([0.5]))
    # m = dist.bernoulli(m, 'n2', 'n1')

    # m = func.condition(m, 'n2', jnp.array([0.9]))

    # proposal = models.create_forward_model()
    # cond = core.const_node('n3', jnp.full((10, )))
    # proposal = dist.beta(proposal, 'n1', cond, jnp.full((10, ), 0.5))
    # proposal = func.proposal(proposal, 'n3')

    # initial_samples = {'n1': jnp.full((10, ), 0.5)}
    # metropolis_hastings_model = func.metropolis_hastings(m,
    #                                                      proposal,
    #                                                      initial_samples,
    #                                                      burnin_steps=100,
    #                                                      num_chains=10)

    # samples = []
    # keys = jax.random.split(jax.random.PRNGKey(123), 100)
    # for i in range(100):
    #     samples.append(metropolis_hastings_model.sample(keys[i]))

    # n1_samples = jnp.stack([i['n1'] for i in samples]).reshape((-1, ))
    # assert jnp.isclose(jnp.mean(n1_samples), 0.9974573)
