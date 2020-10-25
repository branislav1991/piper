# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

from typing import Optional, Union

import jax.random
import jax.numpy as jnp

from piper.distributions import distribution
from piper.functional import kl_divergence
from piper import graph
from piper import util


class Normal(distribution.Distribution):
    def __init__(self, name: str, mu: Union[str, jnp.ndarray],
                 sigma: Union[str, jnp.ndarray]):
        """Initializes a normal distribution with mean mu and standard deviation sigma.

        Mu and sigma may be multidimensional, in which case they represent
        multiple univariate Gaussians.

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

        if isinstance(mu, str):
            self.dependencies.append(mu)
            self.mu = mu
        else:
            self.mu = jnp.array(mu, dtype=jnp.float_)

        if isinstance(sigma, str):
            self.dependencies.append(sigma)
            self.sigma = sigma
        else:
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

        if isinstance(self.mu, str):
            assert self.mu in kwargs
            mu_sample = kwargs[self.mu]
        else:
            mu_sample = self.mu

        if isinstance(self.sigma, str):
            assert self.sigma in kwargs
            sigma_sample = kwargs[self.sigma]
        else:
            sigma_sample = self.sigma

        if mu_sample.shape != sigma_sample.shape:
            raise RuntimeError("Mu and sigma need to be of same shape")

        std_norm = jax.random.normal(key,
                                     shape=mu_sample.shape,
                                     dtype=mu_sample.dtype)

        return std_norm * sigma_sample + mu_sample

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError


def normal(model: graph.Graph, name: str, mu: Union[str, jnp.ndarray],
           sigma: Union[str, jnp.ndarray]):
    if not model:
        raise ValueError('model may not be None')

    dist = Normal(name, mu, sigma)
    model.add(dist)
    return model


@kl_divergence.register_kl(Normal, Normal)
def kl_normal_normal(dist1, dist2):
    if isinstance(dist1.mu, jnp.ndarray) and isinstance(dist2.mu, jnp.ndarray):
        if dist1.mu.shape != dist2.mu.shape:
            raise ValueError('Mu and sigma need to have the same shape')

        k = 1
        return 0.5 * (
            (dist1.sigma / dist2.sigma) + (dist2.mu - dist1.mu)
            * (1. / dist2.sigma)
            * (dist2.mu - dist1.mu) - k + jnp.log(dist2.sigma / dist1.sigma))

    # TODO: we need to measure the kl-divergence in this case by sampling
    raise NotImplementedError
