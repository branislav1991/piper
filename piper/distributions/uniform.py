# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

import jax
import jax.numpy as jnp
import jax.scipy.stats.uniform as jax_uniform

from piper.distributions.distribution import Distribution
from piper import core


class Uniform(Distribution):
    def __init__(self, shape):
        """Initializes a uniform distribution.

        Returns a uniformly distributed value between 0 and 1.

        Args:
            shape: shape of the distribution as a tuple.
        """
        super().__init__()

        self.shape = shape

    def can_condition(self, val: jnp.ndarray):
        in_range = jnp.logical_and(
            jnp.all(jnp.greater_equal(val, jnp.array(0.))),
            jnp.all(jnp.greater_equal(jnp.array(1.), val)))
        return in_range

    def sample(self, key: jnp.ndarray) -> jnp.ndarray:
        """Sample from the distribution.

        Args:
            key: JAX random key.
        """
        return jax.random.uniform(key, shape=self.shape)

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        return jax_uniform.logpdf(x)  # returns 0


def uniform(shape=()):
    return Uniform(shape)


@core.register_kl(Uniform, Uniform)
def kl_normal_normal(dist1: Uniform, dist2: Uniform):
    if dist1.shape != dist2.shape:
        raise ValueError(
            'Cannot calculate KL-divergence with different shapes')

    return jnp.zeros(dist1.shape)
