# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

import abc
from typing import Optional

import jax.numpy as jnp

from piper import graph


class Distribution(graph.Node):
    def __init__(self, name: str):
        super().__init__(name)

    @abc.abstractmethod
    def sample(self, seed: Optional[int], **kwargs):
        """Sample from the distribution.

        Args:
            seed: An optional rng seed. If not specified, the default
                rng seed will be used.
            kwargs: Parameters of the distribution provided as a dictionary.
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
