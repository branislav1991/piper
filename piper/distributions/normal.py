# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

from typing import Optional, Union

import jax.random
import jax.numpy as jnp

from piper.distributions import distribution
from piper import graph
from piper import util


class Normal(distribution.Distribution):
    def __init__(self, name: str, mu: Union[str, jnp.ndarray],
                 sigma: Union[str, jnp.ndarray]):
        """Initializes a normal distribution with mean mu and standard deviation sigma.

        Args:
            mu: Mean of the distribution. This can be either a named entity
                specified in the model or a JAX ndarray.
            sigma: Standard deviation of the distribution. If a JAX ndarray is
                provided, it must have the same dtype and shape as mu.
        """
        super().__init__(name)

        if (not isinstance(mu, jnp.ndarray) and not isinstance(mu, str)):
            raise TypeError('Mu needs to be one of: jnp.ndarray, str')

        if (not isinstance(sigma, jnp.ndarray) and not isinstance(sigma, str)):
            raise TypeError('Sigma needs to be one of: jnp.ndarray, str')

        if isinstance(mu, jnp.ndarray) and isinstance(sigma, jnp.ndarray):
            if mu.shape != sigma.shape:
                raise ValueError('Mu and sigma need to have the same shape')

        self.mu = jnp.array(mu, dtype=jnp.float_)
        self.sigma = jnp.array(sigma, dtype=jnp.float_)

    def sample(self, seed: Optional[int] = None, **kwargs):
        """Sample from the distribution.

        Args:
            seed: An optional rng seed. If not specified, the default
                rng seed will be used.
            kwargs: Parameters of the distribution provided as a dictionary.
        """
        if seed:
            key = jax.random.PRNGKey(seed)
        else:
            key = util.split_default_key()
        return jax.random.normal(key, shape=self.mu.shape, dtype=self.mu.dtype)

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError


def normal(model: graph.Graph, name: str, mu: Union[str, jnp.ndarray],
           sigma: Union[str, jnp.ndarray]):
    if not model:
        raise ValueError('model may not be None')

    dist = Normal(name, mu, sigma)
    model.add(dist)
    return model
