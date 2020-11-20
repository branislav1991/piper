# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

import jax.numpy as jnp

from piper import core


def condition(model: core.Model, node: str, val: jnp.ndarray) -> core.Model:
    """Conditions a node on a variable.

    Args:
        model: Model to apply the conditioning to.
        node: Node to condition on.
        val: Value to condition with.

    Returns:
        New model with the conditioned value.
    """
    if node not in model:
        raise ValueError('Conditioned node not in graph')

    model[node].condition(val)
    return model
