# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

from typing import Union

import jax.random
import jax.numpy as jnp

from piper.distributions import distribution
from piper.functional import kl_divergence
from piper import graph
from piper import param


class Normal(distribution.DistributionNode):
    def __init__(self, name: str, mu: Union[str, jnp.ndarray, param.Param],
                 sigma: Union[str, jnp.ndarray, param.Param]):
        """Initializes a normal distribution with mean mu and standard deviation sigma.

        Mu and sigma may be multidimensional, in which case they represent
        multiple univariate Gaussians.

        Args:
            mu: Mean of the distribution. This can be either a named entity
                specified in the model or a JAX ndarray or a Param.
            sigma: Standard deviation of the distribution. If a concrete value
                is provided, it must have the same dtype and shape as mu.
        """
        super().__init__(name)

        self.mu = param.to_param(mu)
        self.sigma = param.to_param(sigma)

        if isinstance(self.mu, param.FlexibleParam) and \
                isinstance(self.sigma, param.FlexibleParam):
            raise ValueError('Mu and sigma cannot both be flexible')

        if isinstance(self.mu, param.ConstParam) and isinstance(
                self.sigma, param.ConstParam):
            if self.mu.value.shape != self.sigma.value.shape:
                raise ValueError('Mu and sigma need to have the same shape')

        if isinstance(self.mu, param.DependentParam):
            self.dependencies.append(self.mu.name)
        else:
            self.mu.value = self.mu.value.astype(jnp.float32)

        if isinstance(self.sigma, param.DependentParam):
            self.dependencies.append(self.sigma.name)
        else:
            self.sigma.value = self.sigma.value.astype(jnp.float32)

    def sample(self, dependencies: dict, key: jnp.ndarray):
        """Sample from the distribution.

        Args:
            dependencies: dict of dependencies.
            key: JAX random key.
        """
        mu_sample, sigma_sample = distribution._get_samples(
            [self.mu, self.sigma], dependencies)

        if mu_sample.shape != sigma_sample.shape:
            raise RuntimeError("Mu and sigma need to be of same shape")

        if mu_sample.dtype != jnp.float32:
            raise TypeError('mu needs to be jnp.float32')

        if sigma_sample.dtype != jnp.float32:
            raise TypeError('sigma needs to be jnp.float32')

        std_norm = jax.random.normal(key,
                                     shape=mu_sample.shape,
                                     dtype=mu_sample.dtype)

        return std_norm * sigma_sample + mu_sample

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    def _check_valid_condition(self, x: jnp.ndarray):
        if x.dtype != jnp.float32:
            return False

        if (isinstance(self.mu, param.ConstParam)
            and x.shape != self.mu.value.shape) \
            or (isinstance(self.sigma, param.ConstParam)
                and x.shape != self.sigma.value.shape):
            return False

        return True


def normal(model: graph.Graph, name: str, mu: Union[str, jnp.ndarray],
           sigma: Union[str, jnp.ndarray]):
    if not model:
        raise ValueError('model may not be None')

    dist = Normal(name, mu, sigma)
    model.add(dist)
    return model


@kl_divergence.register_kl(Normal, Normal)
def kl_normal_normal(dist1, dist2):
    if isinstance(dist1.mu, param.ConstParam) and isinstance(
            dist2.mu, param.ConstParam):
        if dist1.mu.value.shape != dist2.mu.value.shape:
            raise ValueError('Mu and sigma need to have the same shape')

        mu1 = dist1.mu.value
        mu2 = dist2.mu.value
        sigma1 = dist1.sigma.value
        sigma2 = dist2.sigma.value

        k = 1
        return 0.5 * ((sigma1 / sigma2) + (mu2 - mu1) * (1. / sigma2)
                      * (mu2 - mu1) - k + jnp.log(sigma2 / sigma1))

    # TODO: we need to measure the kl-divergence in this case by sampling
    raise NotImplementedError
