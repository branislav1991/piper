# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

from typing import Union, Dict

import jax
import jax.random
import jax.numpy as jnp
import jax.scipy.stats.beta as jax_beta
import jax.scipy.special as jax_special

from piper.distributions.distribution import Distribution
from piper import core
from piper import utils


class Beta(Distribution):
    def __init__(self, alpha: jnp.ndarray, beta: jnp.ndarray):
        """Initializes a Beta distribution with parameters alpha and beta.

        alpha and beta may be multidimensional, in which case they represent
        multiple Beta distributions.

        Args:
            alpha: Has to be a floating-point type and non-negative.
            beta: Must have the same shape as alpha and be non-negative.
        """
        super().__init__()

        self.alpha = alpha
        self.beta = beta

    def can_condition(self, val: jnp.ndarray):
        return utils.is_floating(val) and jnp.all(0 <= val) \
            and jnp.all(val <= 1)

    def sample(self, key: jnp.ndarray) -> jnp.ndarray:
        """Sample from the distribution.

        Args:
            key: JAX random key.
        """
        shape = self.alpha.shape
        keys = jax.random.split(key, self.alpha.size)
        alpha_sample = self.alpha.reshape((self.alpha.size))
        beta_sample = self.beta.reshape((self.beta.size))
        samp = []
        for a, b, k in zip(alpha_sample, beta_sample, keys):
            samp.append(jax.random.beta(k, a, b))

        return jnp.stack(samp).reshape(shape)

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        """Calculate log probability.
        """
        return jax_beta.logpdf(x, self.alpha, self.beta)


def beta(alpha: jnp.ndarray, beta: jnp.ndarray):
    return Beta(alpha, beta)


@core.register_kl(Beta, Beta)
def kl_beta_beta(dist1: Beta, dist2: Beta):
    a1 = dist1.alpha
    a2 = dist2.alpha
    b1 = dist1.beta
    b2 = dist2.beta

    return jnp.log(jax_special.betainc(a2, b2, 1)
                   / jax_special.betainc(a1, b1, 1)) \
        - (a2 - a1) * jax_special.digamma(a1) \
        - (b2 - b1) * jax_special.digamma(b1) \
        + (a2 - a1 + b2 - b1) * jax_special.digamma(a1 + b1)
