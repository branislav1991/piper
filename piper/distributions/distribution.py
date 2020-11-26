# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

import abc

import jax.numpy as jnp


class Distribution:
    """Parent class for all distributions.

    Allows one to condition on the distribution.
    """
    def __init__(self):
        self._condition = None

    @abc.abstractmethod
    def _can_condition(self, val: jnp.ndarray):
        raise NotImplementedError

    @abc.abstractmethod
    def sample(self, key: jnp.ndarray):
        """Sample from the distribution.

        Args:
            dependencies: Dict of dependencies.
            key: JAX random key.
        """
        if self.is_conditioned():
            return self._condition

        return self._sample(key)

    @abc.abstractmethod
    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        """Returns log probability density for a value.

        Args:
            x: Values to be evaluated. Either a single value or a
                batch of values along dimension 0.

        Returns:
            Log probability of x under the distribution.
        """
        raise NotImplementedError()
