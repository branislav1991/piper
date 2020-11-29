# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

import jax
import jax.numpy as jnp
import jax.scipy.stats.norm as jax_norm

from piper.distributions.distribution import Distribution
from piper import core
from piper import utils


class Normal(Distribution):
    def __init__(self, mu: jnp.ndarray, sigma: jnp.ndarray):
        """Initializes a normal distribution with mean mu and standard deviation sigma.

        Mu and sigma may be multidimensional, in which case they represent
        multiple univariate Gaussians.

        Args:
            mu: Mean of the distribution. This can be either a named entity
                specified in the model or a JAX ndarray or a Param.
            sigma: Standard deviation of the distribution. If a concrete value
                is provided, it must have the same dtype and shape as mu.
        """
        super().__init__()

        if mu.shape != sigma.shape:
            raise ValueError('Mu and sigma need to have the same shape')

        self.mu = mu
        self.sigma = sigma

    def can_condition(self, val: jnp.ndarray):
        return utils.is_floating(val)

    def sample(self, key: jnp.ndarray) -> jnp.ndarray:
        """Sample from the distribution.

        Args:
            key: JAX random key.
        """
        std_norm = jax.random.normal(key,
                                     shape=self.mu.shape,
                                     dtype=self.mu.dtype)

        is_nan = jnp.logical_or(jnp.isnan(self.mu), jnp.isnan(self.sigma))
        return jnp.where(is_nan,
                         jnp.full(self.mu.shape, jnp.nan),
                         std_norm * self.sigma + self.mu)

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        return jax_norm.logpdf(x, self.mu, self.sigma)


def normal(mu: jnp.ndarray, sigma: jnp.ndarray):
    return Normal(mu, sigma)


@core.register_kl(Normal, Normal)
def kl_normal_normal(dist1: Normal, dist2: Normal):
    mu1 = dist1.mu
    mu2 = dist2.mu
    sigma1 = dist1.sigma
    sigma2 = dist2.sigma

    k = 1
    return 0.5 * ((sigma1 / sigma2) + (mu2 - mu1) * (1. / sigma2)
                  * (mu2 - mu1) - k + jnp.log(sigma2 / sigma1))
