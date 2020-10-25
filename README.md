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

    import jax.numpy as jnp
    import piper
    import piper.distributions as dist

    def model():
        alpha0 = jnp.array(10.0)
        beta0 = jnp.array(10.0)
        
        m = piper.create_model()
        m = dist.Beta(m, 'latent_fairness', alpha0, beta0)
        m = dist.Bernoulli(m, f'obs', 'latent_fairness')

        return m
            
This piece of code describes a model to check for the bias of a coin flip. The bias
is modeled as a Beta distribution ("latent_fairness") while the observations
("obs_{i}") are modeled using a Bernoulli distribution. The probability of the 
coin flip being heads ("obs") is given by a sample from the Beta distribution.

After specifying the model, we can sample from it by calling

    sample = piper.sample(model())
    
This will return a dictionary of sampled values.

### Computing KL-divergence

Piper allows you to compute the KL-divergence for defined distributions. This is
embedded in the model API and can be used like this:

    from piper.functional.kl_divergence import kl_divergence
    
    def model():
        m = piper.create_model()
        m = normal(m, 'n1', jnp.array([0., 0.]), jnp.array([1., 1.]))
        m = normal(m, 'n2', jnp.array([1., 1.]), jnp.array([1., 1.]))
        
        return m

    kl_normal_normal = kl_divergence(model(), 'n1', 'n2') # returns 1.0

[JAX]: https://github.com/google/jax