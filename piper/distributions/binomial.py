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
from piper import param


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

        self.n = param.to_param(n)
        self.p = param.to_param(p)

        if isinstance(self.n, param.ConstParam) and isinstance(
                self.p, param.ConstParam):
            if self.n.value.shape != self.p.value.shape:
                raise ValueError('n and p need to have the same shape')

        if isinstance(self.n, param.DependentParam):
            self.dependencies.append(self.n.name)
        else:
            self.n.value = self.n.value.astype(jnp.int32)
            if jnp.any(self.n.value < 0):
                raise ValueError('n must be of type int32 and non-negative')

        if isinstance(self.p, param.DependentParam):
            self.dependencies.append(self.p.name)
        else:
            self.p.value = self.p.value.astype(jnp.float32)
            if jnp.any(self.p.value < 0.) or jnp.any(self.p.value > 1.):
                raise ValueError('p must be of type float32 and in [0, 1]')

    def sample(self, dependencies: dict, key: jnp.ndarray):
        """Sample from the distribution.

        Args:
            dependencies: dict of dependencies.
            key: JAX random key.
        """
        n_sample = self.n.get(dependencies)
        p_sample = self.p.get(dependencies)

        if n_sample.shape != p_sample.shape:
            raise RuntimeError("n and p need to be of same shape")

        assert n_sample.dtype == jnp.int32 and n_sample >= 0
        assert p_sample.dtype == jnp.float32 and (0 <= p_sample <= 1)

        def sample_binomial(n, p, key):
            samples = jax.random.bernoulli(key, p, shape=(n, ))
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


def bernoulli(model: graph.Graph,
              name: str,
              p: Union[str, jnp.ndarray],
              output_shape: Sequence[int] = None):
    if not model:
        raise ValueError('model may not be None')

    n = jnp.ones(p.shape, dtype=jnp.int32)
    dist = Binomial(name, n, p)
    model.add(dist)
    return model


@kl_divergence.register_kl(Binomial, Binomial)
def kl_binomial_binomial(dist1, dist2):
    if isinstance(dist1.n, param.ConstParam) and isinstance(
            dist2.n, param.ConstParam):

        n1 = dist1.n.value
        n2 = dist2.n.value
        p1 = dist1.p.value
        p2 = dist2.p.value

        if jnp.any(n1 != n2):
            raise ValueError('KL-divergence only defined for binomial \
                              distributions with same n')

        if jnp.any(p1 == 0.0) or jnp.any(p1 == 1.0):
            warnings.warn('KL-divergence will be nan', UserWarning)

        if jnp.any(p2 == 0.0) or jnp.any(p2 == 1.0):
            warnings.warn('KL-divergence will be inf', UserWarning)

        return jnp.log(p1 / p2) * n1 * p1 \
            + jnp.log((1. - p1) / (1. - p2)) \
            * n1 * (1. - p1)

    # TODO: we need to measure the kl-divergence in this case by sampling
    raise NotImplementedError
