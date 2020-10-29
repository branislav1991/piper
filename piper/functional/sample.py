# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

import jax.numpy as jnp

from piper import graph
from piper.distributions import distribution


def sample(model: graph.Graph, key: jnp.ndarray) -> graph.Graph:
    """Samples all distributions from the given model.

    Args:
        model: model to sample from.
        key: JAX PRNG key.

    Returns:
        New model with Distribution nodes replaced by ConstNode
        with the sampled value.
    """
    layers = model.topological_sort()
    for layer in layers:
        for node in layer:
            if isinstance(node, distribution.Distribution):
                injected_deps = {}
                for d in node.dependencies:
                    if not isinstance(model[d], graph.ConstNode):
                        raise RuntimeError("Invalid inference on \
                            non-const dependency")

                    injected_deps[d] = model[d].value

                model = graph.replace_node(
                    model, node.name,
                    graph.ConstNode(node.sample(injected_deps, key)))
            else:
                raise TypeError("Unknown node type")

    return model
