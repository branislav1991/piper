# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

from typing import Optional

import jax.random
import jax.numpy as jnp

from piper.distributions import distribution
from piper import util


class Normal(distribution.Distribution):
    def __init__(self, mu: jnp.ndarray,
                 sigma: jnp.ndarray):
        """Initializes a normal distribution with mean mu and standard deviation sigma.

        Args:
            mu: Mean of the distribution as a JAX ndarray.
            sigma: Standard deviation of the distribution. Has to have the
                same type and shape as mu.
        """
        super().__init__()

        assert isinstance(mu, jnp.ndarray) and isinstance(sigma, jnp.ndarray)
        assert mu.shape == sigma.shape, \
            'Mu and sigma need to have the same shape'

        self.mu = jnp.array(mu, dtype=jnp.float_)
        self.sigma = jnp.array(sigma, dtype=jnp.float_)

    def sample(self, seed: Optional[int] = None):
        """Sample from the distribution.

        Args:
            seed: An optional rng seed. If not specified, the default
                rng seed will be used.
        """
        if seed:
            key = jax.random.PRNGKey(seed)
        else:
            key = util.split_default_key()
        return jax.random.normal(key, shape=self.mu.shape, dtype=self.mu.dtype)

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError
