# Piper: Probabilistic Programming for [JAX]

[**Overview**](#overview)
| [**Installation**](#installation)
| [**Tutorial**](#tutorial)

![license](https://img.shields.io/github/license/branislav1991/piper)
![build](https://img.shields.io/github/workflow/status/branislav1991/piper/Python%20package)

## Overview

Piper is a probabilistic programming library supporting variational and sampling inference. Piper is build on [JAX], a numerical computing library that combines NumPy, automatic differentation and GPU/TPU support. For now, Piper only supports sampling from custom models defined by the user. In the future, support for SVI as well as MCMC methods is planned.

## Installation

Piper is written in pure Python. You may simply install piper from pip by running

    pip install --upgrade piper

## Tutorial

### Model Definition

You may define a model in Piper by specifying a generating function like this:

    import jax
    import jax.numpy as jnp
    import piper
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
    sample = piper.sample(model(), key)['obs']
    
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

### MCMC

If you condition on a variable further down the Bayesian network graph, you will
effectively have to sample from the posterior distribution. Trying to do so in the
naive way will result in an exception:

    import piper
    import piper.functional as func

    def model():
        m = models.create_forward_model()
        m = dist.normal(m, 'n1', jnp.array([0.]), jnp.array([1.]))
        m = dist.normal(m, 'n2', 'n1', jnp.array([1.]))
        m = func.condition(m, 'n2', jnp.array([0.5]))
        
        return m
        
    key = jax.random.PRNGKey(123)
    sample = piper.sample(model(), key)['n1']  # will throw a RuntimeError
    
In this case, you will need to rely on a sampling algorithm to obtain a sample from the
posterior. At the moment, piper supports only the Metropolis-Hastings algorithm:

    # With the model defined as above
    proposal = models.create_forward_model()
    proposal = normal(proposal, 'n1' jnp.array([0.]), jnp.array([1.]))
    initial_params = {'n1': 0.}
    mcmc_model = func.mcmc(model(), proposal, initial_params, burn_in_steps=500)
    mcmc_model = func.burnin(mcmc_model)

    samples = []
    keys = jax.random.split(jax.random.PRNGKey(123), 100)
    for i in range(100):  # generate 100 samples after burn-in
        samples.append(piper.sample(mcmc_model, keys[i]))
        
The model returned by *func.metropolis_hastings* will automatically be sampled by the Metropolis-Hastings sampler.

[JAX]: https://github.com/google/jax