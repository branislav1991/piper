# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

from typing import Dict, Callable

import jax.numpy as jnp

from piper.distributions import distribution as dist
from piper.functional.modifier import Modifier


class condition(Modifier):
    def __init__(self, fn: Callable, conditions: Dict[str, jnp.ndarray]):
        """Modifier to condition a model on variables.
        
        Returns NaNs as samples if distribution cannot be conditioned on
        the provided value. This is required so that we do not break the
        paralellization mechanism of Jax. You will be responsible for
        what happens with the sampled values.

        Args:
            conditions: Dictionary of variable names and values
                to condition on.
        """
        super().__init__(fn)
        self.conditions = conditions

    def process(self, node_name: str, d: dist.Distribution):
        if node_name in self.conditions:
            return jnp.where(
                d.can_condition(self.conditions[node_name]),
                self.conditions[node_name],
                jnp.full(self.conditions[node_name].shape, jnp.nan))
        else:
            return None
