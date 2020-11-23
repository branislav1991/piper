# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

from typing import Union, Dict

import jax
import jax.random
import jax.numpy as jnp
import jax.scipy.stats.beta as jax_beta
import jax.scipy.special as jax_special

from piper.functional import kl_divergence
from piper import core
from piper import param


class Beta(core.DistributionNode):
    def __init__(self, name: str, alpha: Union[str, jnp.ndarray],
                 beta: Union[str, jnp.ndarray]):
        """Initializes a Beta distribution with parameters alpha and beta.

        alpha and beta may be multidimensional, in which case they represent
        multiple Beta distributions.

        Args:
            alpha: This can be either a named entity
                specified in the model or a JAX ndarray. If a JAX ndarray
                is provided, it has to be of type float32 and non-negative.
            beta: If a JAX ndarray is provided, it must have the same
                shape and type as alpha and be non-negative.
        """
        super().__init__(name)

        self.alpha = param.to_param(alpha)
        self.beta = param.to_param(beta)

        if isinstance(self.alpha, param.DependentParam):
            self.dependencies.append(self.alpha.name)

        if isinstance(self.beta, param.DependentParam):
            self.dependencies.append(self.beta.name)

    def _sample(self, dependencies: Dict, key: jnp.ndarray) -> jnp.ndarray:
        """Sample from the distribution.

        Args:
            dependencies: Dict of dependencies.
            key: JAX random key.
        """
        alpha_sample, beta_sample = self._get_samples([self.alpha, self.beta],
                                                      dependencies)

        if alpha_sample.shape != beta_sample.shape:
            raise RuntimeError("alpha and beta need to be of same shape")

        shape = alpha_sample.shape
        keys = jax.random.split(key, alpha_sample.size)
        alpha_sample = alpha_sample.reshape((alpha_sample.size))
        beta_sample = beta_sample.reshape((beta_sample.size))
        samp = []
        for a, b, k in zip(alpha_sample, beta_sample, keys):
            samp.append(jax.random.beta(k, a, b))

        return jnp.stack(samp).reshape(shape)

    def _log_prob(self, x: jnp.ndarray, dependencies: Dict) -> jnp.ndarray:
        """Calculate log probability.
        """
        if jnp.any(jnp.logical_or(x < 0, x > 1)):
            raise ValueError('x not in domain of Beta distribution')

        alpha_sample, beta_sample = self._get_samples([self.alpha, self.beta],
                                                      dependencies)
        if alpha_sample.shape != beta_sample.shape:
            raise RuntimeError("alpha and beta need to be of same shape")

        return jax_beta.logpdf(x, alpha_sample, beta_sample)


def beta(model: core.Model, name: str, alpha: Union[str, jnp.ndarray],
         beta: Union[str, jnp.ndarray]):
    if not model:
        raise ValueError('model may not be None')

    dist = Beta(name, alpha, beta)
    model.add(dist)
    return model


@kl_divergence.register_kl(Beta, Beta)
def kl_beta_beta(dist1: Beta, dist2: Beta):
    if isinstance(dist1.alpha, param.ConstParam) and isinstance(
            dist1.beta, param.ConstParam) and isinstance(
                dist2.alpha, param.ConstParam) and isinstance(
                    dist2.beta, param.ConstParam):

        a1 = dist1.alpha.value
        a2 = dist2.alpha.value
        b1 = dist1.beta.value
        b2 = dist2.beta.value

        return jnp.log(jax_special.betainc(a2, b2, 1)
                       / jax_special.betainc(a1, b1, 1)) \
            - (a2 - a1) * jax_special.digamma(a1) \
            - (b2 - b1) * jax_special.digamma(b1) \
            + (a2 - a1 + b2 - b1) * jax_special.digamma(a1 + b1)

    # TODO: we need to measure the kl-divergence in this case by sampling
    raise NotImplementedError
