# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

import abc

import jax.numpy as jnp


class Distribution(abc.ABC):
    @abc.abstractmethod
    def sample(self):
        """Samples a random value.

        Returns:
            Sampled value with same shape as parameters.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        """Returns log probability density for a value.

        Args:
            x: Values to be evaluated. Either a single value or a
                batch of values along dimension 0.

        Returns:
            Log probability of x under the distribution.
        """
        raise NotImplementedError
