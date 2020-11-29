# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

import pytest

import jax
import jax.numpy as jnp

import piper.functional as func
import piper.distributions as dist
from piper import test_util as tu


def test_metropolis_hastings_wrong_proposal_or_initial_samples():
    def model(key):
        n1 = func.sample('n1', dist.normal(jnp.array(0.), jnp.array(1.)), key)
        n2 = func.sample('n2', dist.normal(n1, jnp.array(1.)), key)
        return n2

    def proposal1(key, **current):
        n4 = func.sample('n4', dist.normal(current['n1'], jnp.array(1.)), key)
        return {'n4': n4}

    conditioned_model = func.condition(model, {'n2': jnp.array(0.5)})
    initial_samples = {'n1': jnp.array(0.)}

    with pytest.raises(RuntimeError):  # wrong proposal function
        func.metropolis_hastings(conditioned_model,
                                 proposal1,
                                 initial_samples,
                                 num_chains=1)

    def proposal2(key, **current):
        n1 = func.sample('n1', dist.normal(current['n1'], jnp.array(1.)), key)
        return {'n1': n1}

    initial_samples = {'n3': jnp.array(0.)}

    with pytest.raises(RuntimeError):
        func.metropolis_hastings(conditioned_model,
                                 proposal2,
                                 initial_samples,
                                 num_chains=1)


def test_metropolis_hastings_normal_one_chain():
    def model(key):
        n1 = func.sample('n1', dist.normal(jnp.array(0.), jnp.array(1.)), key)
        return n1

    def proposal(key, **current):
        n1 = func.sample('n1', dist.normal(current['n1'], jnp.array(5.)), key)
        return {'n1': n1}

    initial_samples = {'n1': jnp.array(1.)}

    mcmc_model = func.metropolis_hastings(model,
                                          proposal,
                                          initial_samples,
                                          num_chains=1)

    keys = jax.random.split(jax.random.PRNGKey(123), 300)
    samples = []
    for i in range(300):
        samples.append(mcmc_model(keys[i])['n1'])

    samples = jnp.stack(samples)
    tu.check_close(jnp.mean(samples[200:]), -0.23944321)


def test_metropolis_hastings_normal_normal_one_chain():
    def model(key):
        n1 = func.sample('n1', dist.normal(jnp.array(0.), jnp.array(1.)), key)
        n2 = func.sample('n2', dist.normal(n1, jnp.array(1.)), key)
        return n2

    conditioned_model = func.condition(model, {'n2': jnp.array(0.5)})

    def proposal(key, **current):
        n1 = func.sample('n1', dist.normal(current['n1'], jnp.array(5.)), key)
        return {'n1': n1}

    initial_samples = {'n1': jnp.array(0.)}

    mcmc_model = func.metropolis_hastings(conditioned_model,
                                          proposal,
                                          initial_samples,
                                          num_chains=1)

    keys = jax.random.split(jax.random.PRNGKey(123), 300)
    samples = []
    for i in range(300):
        samples.append(mcmc_model(keys[i])['n1'])

    samples = jnp.stack(samples)
    tu.check_close(jnp.mean(samples[200:]), 0.10883519)


def test_metropolis_hastings_normal_normal_10_chains():
    def model(key):
        n1 = func.sample('n1', dist.normal(jnp.array(0.), jnp.array(1.)), key)
        n2 = func.sample('n2', dist.normal(n1, jnp.array(1.)), key)
        return n2

    conditioned_model = func.condition(model, {'n2': jnp.array(0.5)})

    def proposal(key, **current):
        n1 = func.sample('n1', dist.normal(current['n1'], jnp.array(5.)), key)
        return {'n1': n1}

    initial_samples = {'n1': jnp.array(0.)}

    mcmc_model = func.metropolis_hastings(conditioned_model,
                                          proposal,
                                          initial_samples,
                                          num_chains=10)

    keys = jax.random.split(jax.random.PRNGKey(123), 300)
    samples = []
    for i in range(300):
        samples.append(mcmc_model(keys[i])['n1'])

    samples = jnp.stack(samples)[200:].reshape(-1)
    tu.check_close(jnp.mean(samples), 0.21243224)


def test_metropolis_hastings_normal_normal_multidim_10_chains():
    def model(key):
        n1 = func.sample('n1',
                         dist.normal(jnp.zeros((2, 2)), jnp.full((2, 2), 1.)),
                         key)
        n2 = func.sample('n2', dist.normal(n1, jnp.ones((2, 2))), key)
        return n2

    conditioned_model = func.condition(model, {'n2': jnp.full((2, 2), 0.5)})

    def proposal(key, **current):
        n1 = func.sample('n1',
                         dist.normal(current['n1'], jnp.full((2, 2), 5.)),
                         key)
        return {'n1': n1}

    initial_samples = {'n1': jnp.zeros((2, 2))}

    mcmc_model = func.metropolis_hastings(conditioned_model,
                                          proposal,
                                          initial_samples,
                                          num_chains=10)

    keys = jax.random.split(jax.random.PRNGKey(123), 300)
    samples = []
    for i in range(300):
        samples.append(mcmc_model(keys[i])['n1'])

    samples = jnp.stack(samples)[200:]
    tu.check_close(
        jnp.mean(samples, (0, 1)),
        jnp.array([[0.17168559, 0.14462896], [0.15957117, 0.13937134]]))


def test_metropolis_hastings_beta_bernoulli_10chains():
    def model(key):
        n1 = func.sample('n1', dist.beta(jnp.array(0.5), jnp.array(0.5)), key)
        n2 = func.sample('n2', dist.bernoulli(n1), key)
        return n2

    conditioned_model = func.condition(model, {'n2': jnp.array(1)})

    def proposal(key, **current):  # use prior as proposal
        n1 = func.sample('n1', dist.beta(jnp.array(0.5), jnp.array(0.5)), key)
        return {'n1': n1}

    initial_samples = {'n1': jnp.array(0.5)}

    mcmc_model = func.metropolis_hastings(conditioned_model,
                                          proposal,
                                          initial_samples,
                                          num_chains=10)

    keys = jax.random.split(jax.random.PRNGKey(123), 300)
    samples = []
    for i in range(300):
        samples.append(mcmc_model(keys[i])['n1'])

    samples = jnp.stack(samples)[200:]
    tu.check_close(jnp.mean(samples), 0.999707)
