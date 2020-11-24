# Piper: Probabilistic Programming for [JAX]

[**Overview**](#overview)
| [**Installation**](#installation)
| [**Tutorial**](#tutorial)

![license](https://img.shields.io/github/license/branislav1991/piper)
![build](https://img.shields.io/github/workflow/status/branislav1991/piper/Python%20package)

## Overview

Piper is a probabilistic programming library supporting variational and sampling inference. Piper is build on [JAX], a numerical computing library that combines NumPy, automatic differentation and GPU/TPU support. For now, Piper supports forward sampling from custom models defined by the user as well as the Metropolis-Hastings method. In the future, support for Hamiltonian Monte Carlo as well as SVI is planned.

## Installation

Piper is written in pure Python. You may simply install piper from pip by running

    pip install --upgrade piper

## Tutorial

### Model Definition

You may define a model in Piper by specifying a generating function like this:

    import jax
    import jax.numpy as jnp
    import piper.distributions as dist
    import piper.models as models

    def model():
        alpha0 = jnp.array(10.0)
        beta0 = jnp.array(10.0)
        
        m = models.create_forward_model()
        m = dist.beta(m, 'latent_fairness', alpha0, beta0)
        m = dist.bernoulli(m, 'obs', 'latent_fairness')

        return m
            
This piece of code describes a model to check for the bias of a coin flip. The bias
is modeled as a Beta distribution ("latent_fairness") while the observations
("obs_{i}") are modeled using a Bernoulli distribution. The probability of the 
coin flip being heads ("obs") is given by a sample from the Beta distribution.

After specifying the model, we can sample from it by calling

    key = jax.random.PRNGKey(123)
    sample = model.sample(key)['obs']
    
This will return a dictionary of sampled values.

### Computing KL-divergence

Piper allows you to compute the KL-divergence for defined distributions. This is
embedded in the model API and can be used like this:

    import piper.functional as func
    
    def model():
        m = models.create_forward_model()
        m = dist.normal(m, 'n1', jnp.array([0., 0.]), jnp.array([1., 1.]))
        m = dist.normal(m, 'n2', jnp.array([1., 1.]), jnp.array([1., 1.]))
        
        return m

    kl_normal_normal = func.compute_kl_div(model(), 'n1', 'n2') # returns [0.5, 0.5]
    
### Conditioning
    
Conditioning on Bayesian network variables is easy:

    import piper.functional as func

    def model():
        m = models.create_forward_model()
        m = dist.normal(m, 'n1', jnp.array([0.]), jnp.array([1.]))
        m = dist.normal(m, 'n2', 'n1', jnp.array([1.]))
        m = func.condition(m, 'n1', jnp.array([0.5]))
        
        return m
        
You may now sample from the conditional distribution.

### Metropolis-Hastings

If you condition on a variable further down the Bayesian network graph, you will
effectively have to sample from the posterior distribution. Trying to do so in the
naive way will result in an exception:

    import piper.functional as func
    from piper import core

    def model():
        m = models.create_forward_model()
        m = dist.normal(m, 'n1', jnp.array([0.]), jnp.array([1.]))
        m = dist.normal(m, 'n2', 'n1', jnp.array([1.]))
        m = func.condition(m, 'n2', jnp.array([0.5]))
        
        return m
        
    key = jax.random.PRNGKey(123)
    sample = model.sample(key)['n1']  # will throw a RuntimeError
    
In this case, you will need to rely on a sampling algorithm to obtain a sample from the
posterior. At the moment, piper supports only the Metropolis-Hastings algorithm:

    # With the model defined as above
    proposal = models.create_forward_model()
    proposal = core.const_node(proposal, 'n3', jnp.array([0.]))
    proposal = normal(proposal, 'n1' jnp.array([0.]), jnp.array([1.]))
    proposal = func.proposal(proposal, 'n3')
    
    initial_samples = {'n1': jnp.array([0.])}
    metropolis_hastings_model = func.metropolis_hastings(model(), proposal, initial_samples, burn_in_steps=500, num_chains=1)

    samples = []
    keys = jax.random.split(jax.random.PRNGKey(123), 100)
    for i in range(100):  # generate 100 samples after burn-in
        samples.append(metropolis_hastings_model.sample(keys[i]))
        
First we create a proposal distribution to propose samples for us. This is done using *func.proposal*. As the proposal proposes
new samples like *P(x'|x)*, we need to specify which node will be used as *x* for conditioning. This is done using the second
parameter of *func.proposal*.

The model returned by *func.metropolis_hastings* will automatically be sampled by the Metropolis-Hastings sampler. Note that using multiple chains will be
automatically parallelized by piper.

[JAX]: https://github.com/google/jax