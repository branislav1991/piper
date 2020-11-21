# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

from typing import Union, Dict
import math

import jax.random
import jax.numpy as jnp

from piper.functional import kl_divergence
from piper import core
from piper import param


class Normal(core.DistributionNode):
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

        if isinstance(self.mu, param.DependentParam):
            self.dependencies.append(self.mu.name)

        if isinstance(self.sigma, param.DependentParam):
            self.dependencies.append(self.sigma.name)

    def _sample(self, dependencies: Dict, key: jnp.ndarray):
        """Sample from the distribution.

        Args:
            dependencies: Dict of dependencies.
            key: JAX random key.
        """
        mu_sample, sigma_sample = self._get_samples([self.mu, self.sigma],
                                                    dependencies)

        if mu_sample.shape != sigma_sample.shape:
            raise RuntimeError("Mu and sigma need to be of same shape")

        std_norm = jax.random.normal(key,
                                     shape=mu_sample.shape,
                                     dtype=mu_sample.dtype)

        return std_norm * sigma_sample + mu_sample

    def _log_prob(self, x: jnp.ndarray, dependencies: Dict) -> jnp.ndarray:
        mu_sample, sigma_sample = self._get_samples([self.mu, self.sigma],
                                                    dependencies)

        return (-jnp.log(sigma_sample) - 0.5 * jnp.log(2 * math.pi) - 0.5
                * ((x - mu_sample) / sigma_sample)**2)


def normal(model: core.Model, name: str, mu: Union[str, jnp.ndarray],
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

        mu1 = dist1.mu.value
        mu2 = dist2.mu.value
        sigma1 = dist1.sigma.value
        sigma2 = dist2.sigma.value

        k = 1
        return 0.5 * ((sigma1 / sigma2) + (mu2 - mu1) * (1. / sigma2)
                      * (mu2 - mu1) - k + jnp.log(sigma2 / sigma1))

    # TODO: we need to measure the kl-divergence in this case by sampling
    raise NotImplementedError
