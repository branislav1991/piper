# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

import jax
import jax.numpy as jnp

from piper.distributions.distribution import Distribution
from piper.functional import kl_divergence
from piper import core
from piper import utils


class Binomial(Distribution):
    def __init__(self, n: jnp.ndarray, p: jnp.ndarray):
        """Initializes a binomial distribution with n trials and probability p.

        n and p may be multidimensional, in which case they represent
        multiple binomial distributions.

        Args:
            n: Number of trials. Has to be an integer and non-negative.
            p: Probability of the success of a trial. Must have same shape
                as n and must be between 0 and 1.
        """
        super().__init__()

        if n.shape != p.shape:
            raise ValueError('n and p need to have the same shape')

        self.n = jnp.int32(n)
        self.p = p

    def can_condition(self, val: jnp.ndarray):
        return utils.is_integer(val)

    def sample(self, key: jnp.ndarray) -> jnp.ndarray:
        """Sample from the distribution.

        Args:
            key: JAX random key.
        """
        shape = self.n.shape
        keys = jax.random.split(key, self.n.size)
        n_sample = self.n.reshape((self.n.size))
        p_sample = self.p.reshape((self.p.size))
        samp = []
        for n, p, k in zip(n_sample, p_sample, keys):
            samples = jax.random.bernoulli(k, p, shape=(n, ))
            samp.append(jnp.sum(samples))

        is_nan = jnp.isnan(self.p)
        return jnp.where(is_nan, jnp.full(shape, jnp.nan),
                         jnp.stack(samp).reshape(shape))

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        """Calculate log probability.
        """
        return x * jnp.log(self.p) + (self.n - x) * jnp.log(1 - self.p)


def binomial(n: jnp.ndarray, p: jnp.ndarray):
    return Binomial(n, p)


def bernoulli(p: jnp.ndarray):
    n = jnp.ones_like(p, dtype=jnp.int32)
    return Binomial(n, p)


@core.register_kl(Binomial, Binomial)
def kl_binomial_binomial(dist1: Binomial, dist2: Binomial):
    n1 = dist1.n
    n2 = dist2.n
    p1 = dist1.p
    p2 = dist2.p

    if jnp.any(n1 != n2):
        raise ValueError('KL-divergence only defined for binomial \
                          distributions with same n')

    return jnp.log(p1 / p2) * n1 * p1 \
        + jnp.log((1. - p1) / (1. - p2)) \
        * n1 * (1. - p1)
