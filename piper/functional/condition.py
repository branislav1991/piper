# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

from typing import Dict

import jax.numpy as jnp

from piper import core
from piper.distributions import distribution as dist


class Condition:
    def __init__(self, conditions: Dict[str, jnp.ndarray]):
        """Context manager to condition a model on variables.

        Args:
            conditions: Dictionary of variable names and values
                to condition on.
        """
        self.conditions = conditions

    def __enter__(self):
        core._MODIFIER_STACK.append(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert core._MODIFIER_STACK[-1] == self
        core._MODIFIER_STACK.pop()

    def process(self, node_name: str, d: dist.Distribution, key: jnp.ndarray):
        if node_name in self.conditions:
            if not d.can_condition(self.conditions[node_name]):
                raise ValueError('Cannot condition on this variable')

            return self.conditions[node_name]
        else:
            return None
