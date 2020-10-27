# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

from typing import Union, Sequence
import warnings

import jax
import jax.random
import jax.numpy as jnp

from piper.distributions import distribution
from piper.functional import kl_divergence
from piper import graph


class Binomial(distribution.Distribution):
    def __init__(self, name: str, n: Union[str, jnp.ndarray],
                 p: Union[str, jnp.ndarray]):
        """Initializes a binomial distribution with n trials and probability p.

        n and p may be multidimensional, in which case they represent
        multiple binomial distributions.

        Args:
            n: Number of trials. This can be either a named entity
                specified in the model or a JAX ndarray. If a JAX ndarray
                is provided, it has to be of type int32 and non-negative.
            p: Probability of the success of a trial. If a JAX ndarray is
                provided, it must have the same shape as n. If must be
                between 0 and 1.
        """
        super().__init__(name)

        if (not isinstance(n, jnp.ndarray) and not isinstance(n, str)):
            raise TypeError('n needs to be one of: jnp.ndarray, str')

        if (not isinstance(p, jnp.ndarray) and not isinstance(p, str)):
            raise TypeError('p needs to be one of: jnp.ndarray, str')

        if isinstance(n, jnp.ndarray) and isinstance(p, jnp.ndarray):
            if n.shape != p.shape:
                raise ValueError('n and p need to have the same shape')

        if isinstance(n, str):
            self.dependencies.append(n)
            self.n = n
        else:
            if n.dtype != jnp.int32 or jnp.any(n < 0):
                raise ValueError('n must be of type int32 and non-negative')
            self.n = n

        if isinstance(p, str):
            self.dependencies.append(p)
            self.p = p
        else:
            if jnp.any(p < 0.) or jnp.any(p > 1.):
                raise ValueError('p must be in the interval [0,1]')
            self.p = p

    def sample(self, key: jnp.ndarray, **kwargs):
        """Sample from the distribution.

        Args:
            key: JAX random key.
            kwargs: Parameters of the distribution provided as a dictionary.
        """
        if isinstance(self.n, str):
            assert self.n in kwargs and kwargs[self.n].dtype == jnp.int32 \
                and kwargs[self.n] >= 0
            n_sample = kwargs[self.n]
        else:
            n_sample = self.n

        if isinstance(self.p, str):
            assert self.p in kwargs and (0 <= kwargs[self.p] <= 1)
            p_sample = kwargs[self.p]
        else:
            p_sample = self.p

        if n_sample.shape != p_sample.shape:
            raise RuntimeError("n and p need to be of same shape")

        def sample_binomial(n, p, key):
            samples = jax.random.bernoulli(key, p, shape=(n,))
            return jnp.sum(samples)

        shape = n_sample.shape
        keys = jax.random.split(key, n_sample.size)
        n_sample = n_sample.reshape((n_sample.size))
        p_sample = p_sample.reshape((p_sample.size))
        samp = []
        for n, p, k in zip(n_sample, p_sample, keys):
            samp.append(sample_binomial(n, p, k))

        return jnp.stack(samp).reshape(shape)

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError


def binomial(model: graph.Graph, name: str, n: Union[str, jnp.ndarray],
             p: Union[str, jnp.ndarray]):
    if not model:
        raise ValueError('model may not be None')

    dist = Binomial(name, n, p)
    model.add(dist)
    return model


def bernoulli(model: graph.Graph, name: str, p: Union[str, jnp.ndarray],
              output_shape: Sequence[int] = None):
    if not model:
        raise ValueError('model may not be None')

    n = jnp.ones(p.shape, dtype=jnp.int32)
    dist = Binomial(name, n, p)
    model.add(dist)
    return model


@kl_divergence.register_kl(Binomial, Binomial)
def kl_binomial_binomial(dist1, dist2):
    if isinstance(dist1.n, jnp.ndarray) and isinstance(dist2.n, jnp.ndarray):
        if jnp.any(dist1.n != dist2.n):
            raise ValueError('KL-divergence only defined for binomial \
                              distributions with same n')

        if jnp.any(dist2.p == 0.0) or jnp.any(dist2.p == 1.0):
            warnings.warn('KL-divergence will be inf', UserWarning)

        if jnp.any(dist1.p == 0.0) or jnp.any(dist1.p == 1.0):
            warnings.warn('KL-divergence will be nan', UserWarning)

        return jnp.log(dist1.p / dist2.p) * dist1.n * dist1.p \
            + jnp.log((1. - dist1.p) / (1. - dist2.p)) \
            * dist1.n * (1. - dist1.p)

    # TODO: we need to measure the kl-divergence in this case by sampling
    raise NotImplementedError
