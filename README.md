# Piper: Probabilistic Programming for [JAX]

[**Overview**](#overview)
| [**Installation**](#installation)
| [**Tutorial**](#tutorial)

![license](https://img.shields.io/github/license/branislav1991/piper)
![Publish Package](https://github.com/branislav1991/piper/workflows/Publish%20Package/badge.svg)
![Build Python Package](https://github.com/branislav1991/piper/workflows/Build%20Python%20Package/badge.svg)

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
    import piper.functional as func

    def model():
        alpha0 = jnp.array(0.5)
        beta0 = jnp.array(0.5)
        
        keys = jax.random.split(jax.random.PRNGKey(123), 2)
        
        fairness = func.sample('latent_fairness', dist.beta(alpha0, beta0), keys[0])
        obs = func.sample('obs', dist.bernoulli(fairness), keys[1])

        return [fairness, obs]
            
This piece of code describes a model to check for the bias of a coin flip. The bias
is modeled as a Beta distribution ("latent_fairness") while the observations
("obs_{i}") are modeled using a Bernoulli distribution. The probability of the 
coin flip being heads ("obs") is given by a sample from the Beta distribution.

After specifying the model, we can sample from it by calling

    sample = model()

### Computing KL-divergence

Piper allows you to compute the KL-divergence for defined distributions. This is
embedded in the model API and can be used like this:

    n1 = dist.normal('n1', jnp.array([0., 0.]), jnp.array([1., 1.]))
    n2 = dist.normal('n2', jnp.array([1., 1.]), jnp.array([1., 1.]))
        
    kl_normal_normal = func.compute_kl_div(n1, n2) # returns [0.5, 0.5]
    
### Conditioning
    
Conditioning on Bayesian network variables is performed by enclosing the sampling
procedure in a special *Condition* context:

    def model():
        keys = jax.random.split(jax.random.PRNGKey(123), 2)

        n1 = func.sample('n1', dist.normal(jnp.array(0.), jnp.array(1.)), keys[0])
        n2 = func.sample('n2', dist.normal(n1, jnp.array(1.)), keys[1])

        return n2
        
    conditioned_model = func.condition(model, {'n1': jnp.array(0.)})
    sample = conditioned_model()
        
You may now sample from the conditional distribution by calling it as you would an unconditioned model:

### Metropolis-Hastings

If you condition on a variable further down the Bayesian network graph, you will
effectively have to sample from the posterior distribution. Trying to do so in the
naive way won't capture the posterior distribution:

    def model():
        keys = jax.random.split(jax.random.PRNGKey(123), 2)

        n1 = func.sample('n1', dist.normal(jnp.array(0.), jnp.array(1.)), keys[0])
        n2 = func.sample('n2', dist.normal(n1, jnp.array(1.)), keys[1])
        
        return n1
        
    conditioned_model = func.condition(model, {'n2': jnp.array(1.)})
    sample = conditioned_model()  # will give incorrect result
 

In this case, you will need to rely on a sampling algorithm to obtain a sample from the
posterior. At the moment, piper supports the Metropolis-Hastings algorithm:

    # With the model defined as above
    def proposal(key, **current_samples):
        proposal_n1 = func.sample('n1', dist.normal(current_samples['n1'], jnp.array(5.)), key)
        return {'n1': proposal_n1}
    
    initial_samples = {'n1': jnp.array(0.)}
    metropolis_hastings_model = func.metropolis_hastings(conditioned_model, 
                                                         proposal, 
                                                         initial_samples, 
                                                         num_chains=1)

    samples = []
    keys = jax.random.split(jax.random.PRNGKey(123), 500)
    for i in range(500):
        sample = metropolis_hastings_model(keys[i])
        if i >= 100:  # ignore first 100 samples as burn-in
            samples.append(sample)
        
First we create a proposal model to propose samples for us. As the proposal proposes
new samples like *P(x'|x)*, we need to specify which node will be used as *x* for conditioning.
This is done using the parameters of *proposal*. Each parameter in the *current_samples* dictionary
is named and assigned a conditioning value.

The model returned by *func.metropolis_hastings* will automatically be sampled by the
Metropolis-Hastings sampler. Note that using multiple chains will be
automatically parallelized by piper.

### Stochastic Variational Inference

Another way to sample from a complicated distribution is to perform Stochastic Variational Inference (SVI).
In SVI, instead of running a Markov chain, we use an auxiliary distribution *q* with
an arbitrary number of parameters and we try to adjust these parameters so that *q* approximates our original distribution *p*. In particular, we choose *q* so that we can
easily sample from it.

Finding the correct parameters of *q* is done using gradient descent on our data. Let us demonstrate this using a very simple example. In this example, we will try to find the correct parameters to approximate a normal distribution with a normal distribution.
Since these distributions are qualitatively equal, the final parameters after the optimization procedure should match the original ones very closely.


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
    
In this example, we use the ADAM optimizer provided by JAX to infer the distributional
parameters. Running this code, we note that *inferred_n1_mean* and *inferred_n1_std* are very close to 10, our true parameters.

[JAX]: https://github.com/google/jax