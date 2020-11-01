# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

import jax.numpy as jnp

from piper import graph
from piper.distributions import distribution


def condition(model: graph.Graph, node: str, val: jnp.ndarray) -> graph.Graph:
    """Conditions a node on a variable.

    Args:
        model: Model to apply the conditioning to.
        node: Node to condition on.
        val: Value to condition with.

    Returns:
        New mode with ConditionedNode with the conditioned value.
    """
    if node not in model:
        raise ValueError('Conditioned node not in graph')

    model = graph.replace_node(model, node,
                               distribution.conditioned_node(model[node], val))
    return model
