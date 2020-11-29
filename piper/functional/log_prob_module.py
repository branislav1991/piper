# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

from typing import Callable

import jax.numpy as jnp

from piper.distributions import distribution as dist
from piper.functional.modifier import Modifier


class log_prob(Modifier):
    def __init__(self, fn: Callable):
        """Modifier to calculate log probability.

        Calculates the log probability of a stochastic function
        according to the log probabilities of individual
        distributions and sampled values.

        If you want to calculate the log probability of specific
        values under the model, condition on them appropriately.
        """
        super().__init__(fn)
        self.log_prob = 0

    def __enter__(self):
        super().__enter__()
        return self

    def post_process(self, sample: jnp.ndarray, node_name: str,
                     d: dist.Distribution):
        self.log_prob += d.log_prob(sample)

    def get(self):
        """Return calculated log probability.
        """
        return self.log_prob
