# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

import jax
import jax.numpy as jnp
import jax.scipy.stats.dirichlet as jax_dir
import jax.scipy.special as jax_special

from piper.distributions.distribution import Distribution
from piper import core
from piper import utils


class Dirichlet(Distribution):
    def __init__(self, alphas: jnp.ndarray):
        """Initializes a Dirichlet distribution with parameters alphas.

        Args:
            alphas: Has to be a floating-point type and non-negative.
                Last dimension is assumed to be a list of alpha parameters
                for every distribution.
        """
        super().__init__()

        nans = jnp.full(alphas.shape, jnp.nan)
        self.alphas = jnp.where(jnp.any(alphas <= 0, axis=-1, keepdims=True),
                                nans, alphas)

    def can_condition(self, val: jnp.ndarray):
        correct_shape = val.shape == self.alphas.shape
        between_0_and_1 = jnp.logical_and(val >= 0, val <= 1)
        sums_to_1 = jnp.isclose(jnp.sum(val, -1, keepdims=True), 1)
        return jnp.logical_and(
            utils.is_floating(val),
            jnp.logical_and(correct_shape,
                            jnp.logical_and(between_0_and_1, sums_to_1)))

    def sample(self, key: jnp.ndarray) -> jnp.ndarray:
        """Sample from the distribution.

        Args:
            key: JAX random key.
        """
        sample = jax.random.dirichlet(key, self.alphas, self.alphas.shape[:-1])

        is_nan = jnp.isnan(self.alphas)
        return jnp.where(is_nan, jnp.full(self.alphas.shape, jnp.nan), sample)

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        """Calculate log probability.
        """
        return jax_dir.logpdf(x, self.alphas)


def dirichlet(alphas: jnp.ndarray):
    return Dirichlet(alphas)


@core.register_kl(Dirichlet, Dirichlet)
def kl_dir_dir(dist1: Dirichlet, dist2: Dirichlet):
    a1 = dist1.alphas
    a2 = dist2.alphas

    if a1.shape != a2.shape:
        raise ValueError('KL-divergence only defined for Dirichlet \
                          distributions with same dimensions of alphas')

    a_0_1 = jnp.sum(a1, -1, keepdims=True)
    a_0_2 = jnp.sum(a2, -1, keepdims=True)

    kl_div = jax_special.gammaln(a_0_1) \
        - jnp.sum(jax_special.gammaln(a1), -1, keepdims=True) \
        - jax_special.gammaln(a_0_2) \
        + jnp.sum(jax_special.gammaln(a2), -1, keepdims=True) \
        + jnp.sum((a1 - a2) * (
            jax_special.digamma(a1) - jax_special.digamma(a_0_1)),
            -1, keepdims=True)

    return jnp.squeeze(kl_div, -1)
