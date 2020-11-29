# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

from typing import Callable

import jax.numpy as jnp

from piper import core
from piper.distributions import distribution as dist


class Modifier:
    """Modifies the execution of a stochastic function.
    """
    def __init__(self, fn: Callable):
        self.fn = fn

    def __enter__(self):
        core._MODIFIER_STACK.append(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert core._MODIFIER_STACK[-1] == self
        core._MODIFIER_STACK.pop()

    def process(self, node_name: str, d: dist.Distribution):
        pass

    def post_process(self, sample: jnp.ndarray, node_name: str,
                     d: dist.Distribution):
        pass

    def __call__(self, *args, **kwargs):
        with self:
            return self.fn(*args, **kwargs)
