# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln

from piper.distributions.distribution import Distribution
from piper import core
from piper import utils


class Multinomial(Distribution):
    def __init__(self, n: jnp.ndarray, p: jnp.ndarray):
        """Initializes a multinomial distribution with n trials and probabilities p.

        n may be multidimensional, in which case it represents
        multiple multinomial distributions.

        p has to have the shape of n plus 1 dimension representing the
        the probabilities of each event. The probabilities in the last
        dimension have to sum to 1.

        Args:
            n: Number of trials. Has to be an integer and non-negative.
            p: Probabilities of trial successes. Must have same shape
                as n + 1 additional dimension representing the probabilities.
                Probabilities have to sum to 1.
        """
        super().__init__()

        if n.shape != p.shape[:len(n.shape)] or \
                len(n.shape) + 1 != len(p.shape):
            raise ValueError('Shapes of n and p not compatible')

        # we cannot raise a ValueError here since we get problems with
        # ConcretizationError during Metropolis-Hastings
        nans = jnp.full(p.shape, jnp.nan)
        self.p = jnp.where(jnp.logical_or(p < 0, p > 1), nans, p)
        self.p = jnp.where(jnp.isclose(jnp.sum(self.p, -1, keepdims=True), 1),
                           self.p,
                           nans)

        nans = jnp.full(n.shape, jnp.nan)
        self.n = jnp.where(n <= 0, nans, n)

    def can_condition(self, val: jnp.ndarray):
        in_range = jnp.logical_and(
            jnp.all(jnp.greater_equal(val, jnp.array(0.))),
            jnp.all(jnp.greater_equal(self.n, val)))
        correct_size = val.shape == self.p.shape

        return jnp.logical_and(
            jnp.logical_and(utils.is_integer(val), correct_size), in_range)

    def sample(self, key: jnp.ndarray) -> jnp.ndarray:
        """Sample from the distribution.

        Args:
            key: JAX random key.
        """
        shape = self.p.shape
        keys = jax.random.split(key, self.n.size)
        n_sample = self.n.reshape((self.n.size))
        p_sample = self.p.reshape((-1, self.p.shape[-1]))
        samp = []
        for n, p, k in zip(n_sample, p_sample, keys):
            samples = jnp.where(
                jnp.isnan(n), jnp.nan,
                jax.random.categorical(k, jnp.log(p), shape=(jnp.int32(n), )))
            samp.append(jnp.sum(jax.nn.one_hot(samples, p.shape[-1]), 0))

        is_nan = jnp.isnan(self.p)
        return jnp.where(is_nan, jnp.full(shape, jnp.nan),
                         jnp.stack(samp).reshape(shape))

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        """Calculate log probability.
        """
        def log_multinomial_coeff(n, x):
            return gammaln(n + 1) - jnp.sum(gammaln(x + 1), -1)

        return log_multinomial_coeff(self.n, x) + jnp.sum(
            x * jnp.log(self.p), -1)


def multinomial(n: jnp.ndarray, p: jnp.ndarray):
    return Multinomial(n, p)


def categorical(p: jnp.ndarray):
    n = jnp.ones(p.shape[:-1], dtype=jnp.int32)
    return Multinomial(n, p)


@core.register_kl(Multinomial, Multinomial)
def kl_multinomial_multinomial(dist1: Multinomial, dist2: Multinomial):
    n1 = dist1.n
    n2 = dist2.n
    p1 = dist1.p
    p2 = dist2.p
    k = p1.shape[-1]

    if jnp.any(n1 != n2):
        raise ValueError('KL-divergence only defined for multinomial \
                          distributions with same n')

    if p1.shape != p2.shape:
        raise ValueError('KL-divergence only defined for multinomial \
                          distributions with same dimensions of p')

    return n1 * jnp.sum(p1 * jnp.log(p1 / p2), -1)
