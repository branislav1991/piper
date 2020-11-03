# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

import jax.numpy as jnp

from piper import core


def sample(model: core.Model, key: jnp.ndarray) -> core.Model:
    """Samples all dists from the given model.

    Args:
        model: Model to sample from.
        key: JAX PRNG key.

    Returns:
        New model with dist nodes replaced by ConstNode
        with the sampled value.
    """
    if not model.can_sample():
        raise RuntimeError("Invalid forward sampling on \
            conditioned dependency")

    layers = model.topological_sort()
    for layer in layers:
        for node in layer:
            if isinstance(node, core.DistributionNode):
                injected_deps = {}
                for d in node.dependencies:
                    if not isinstance(model[d], core.ConstNode):
                        raise RuntimeError("Invalid forward sampling on \
                            non-const dependency")

                    injected_deps[d] = model[d].value

                model = core.replace_node(
                    model, node.name,
                    core.ConstNode(node.name, node.sample(injected_deps, key)))

            elif isinstance(node, core.ConditionedNode):
                model = core.replace_node(
                    model, node.name, core.ConstNode(node.name, node.value))

            else:
                raise TypeError("Unknown node type")

    return model
