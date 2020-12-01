# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

import jax
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

        if alpha.shape != beta.shape:
            raise ValueError('alpha and beta need to have the same shape')

        nans = jnp.full(alpha.shape, jnp.nan)
        self.alpha = jnp.where(alpha <= 0, nans, alpha)
        self.beta = jnp.where(beta <= 0, nans, beta)

    def can_condition(self, val: jnp.ndarray):
        in_range = jnp.logical_and(
            jnp.all(jnp.greater_equal(val, jnp.array(0.))),
            jnp.all(jnp.greater_equal(jnp.array(1.), val)))
        return jnp.logical_and(utils.is_floating(val), in_range)

    def sample(self, key: jnp.ndarray) -> jnp.ndarray:
        """Sample from the distribution.

        Args:
            key: JAX random key.
        """
        sample = jax.random.beta(key, self.alpha, self.beta)

        is_nan = jnp.logical_or(jnp.isnan(self.alpha), jnp.isnan(self.beta))
        return jnp.where(is_nan,
                         jnp.full(self.alpha.shape, jnp.nan),
                         sample)

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

    s1 = a1 + b1
    s2 = a2 + b2

    kl_div = jax_special.gammaln(s1) \
        - jax_special.gammaln(a1) \
        - jax_special.gammaln(b1) \
        - jax_special.gammaln(s2) \
        + jax_special.gammaln(a2) \
        + jax_special.gammaln(b2) \
        + (a1 - a2) * (jax_special.digamma(a1 - jax_special.digamma(s1))) \
        + (b1 - b2) * (jax_special.digamma(b1 - jax_special.digamma(s1)))

    return kl_div
