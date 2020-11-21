# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

from typing import Union, Dict

import jax
import jax.random
import jax.numpy as jnp

from piper.functional import kl_divergence
from piper import core
from piper import param


class Binomial(core.DistributionNode):
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

        if isinstance(self.n, param.DependentParam):
            self.dependencies.append(self.n.name)

        if isinstance(self.p, param.DependentParam):
            self.dependencies.append(self.p.name)

        def sample_binomial(n, p, key):
            samples = jax.random.bernoulli(key, p, shape=(n, ))
            return jnp.sum(samples)

        self.sample_binomial = jax.jit(sample_binomial, static_argnums=0)

    def _sample(self, dependencies: Dict, key: jnp.ndarray) -> jnp.ndarray:
        """Sample from the distribution.

        Args:
            dependencies: Dict of dependencies.
            key: JAX random key.
        """
        n_sample, p_sample = self._get_samples([self.n, self.p], dependencies)

        if n_sample.shape != p_sample.shape:
            raise RuntimeError("n and p need to be of same shape")

        shape = n_sample.shape
        keys = jax.random.split(key, n_sample.size)
        n_sample = n_sample.reshape((n_sample.size))
        p_sample = p_sample.reshape((p_sample.size))
        samp = []
        for n, p, k in zip(n_sample, p_sample, keys):
            samp.append(self.sample_binomial(n, p, k))

        return jnp.stack(samp).reshape(shape)

    def _log_prob(self, x: jnp.ndarray, dependencies: Dict) -> jnp.ndarray:
        """Calculate log probability.
        """
        raise NotImplementedError


def binomial(model: core.Model, name: str, n: Union[str, jnp.ndarray],
             p: Union[str, jnp.ndarray]):
    if not model:
        raise ValueError('model may not be None')

    dist = Binomial(name, n, p)
    model.add(dist)
    return model


def bernoulli(model: core.Model, name: str, p: Union[str, jnp.ndarray]):
    if not model:
        raise ValueError('model may not be None')

    n = param.flexible_param(jnp.array(1, dtype=jnp.int32))
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

        return jnp.log(p1 / p2) * n1 * p1 \
            + jnp.log((1. - p1) / (1. - p2)) \
            * n1 * (1. - p1)

    # TODO: we need to measure the kl-divergence in this case by sampling
    raise NotImplementedError
