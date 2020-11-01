# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

import collections

import jax.numpy as jnp

from piper import graph
from piper.distributions import distribution


def _check_forward_sampling_possible(model: graph.Graph):
    """Checks if you can apply forward sampling to the model.

    Returns:
        True if there is no conditioning on downstream nodes which have
        non-conditioned and non-const dependencies (otherwise you have
        to estimate posterior).
        False otherwise.
    """
    visited = set()
    queue = collections.deque()
    for node in model.nodes.values():
        if isinstance(node, distribution.ConditionedNode):
            queue.append(node)

    while queue:
        node = queue.popleft()
        if node in visited:
            continue

        visited.add(node)
        if (not isinstance(node, distribution.ConditionedNode)) \
                and (not isinstance(node, graph.ConstNode)):
            return False

        for d in node.dependencies:
            queue.append(model[d])

    return True


def sample(model: graph.Graph, key: jnp.ndarray) -> graph.Graph:
    """Samples all distributions from the given model.

    Args:
        model: Model to sample from.
        key: JAX PRNG key.

    Returns:
        New model with Distribution nodes replaced by ConstNode
        with the sampled value.
    """
    if not _check_forward_sampling_possible(model):
        raise RuntimeError("Invalid forward sampling on \
            conditioned dependency")

    layers = model.topological_sort()
    for layer in layers:
        for node in layer:
            if isinstance(node, distribution.DistributionNode):
                injected_deps = {}
                for d in node.dependencies:
                    if not isinstance(model[d], graph.ConstNode):
                        raise RuntimeError("Invalid forward sampling on \
                            non-const dependency")

                    injected_deps[d] = model[d].value

                model = graph.replace_node(
                    model, node.name,
                    graph.ConstNode(node.name, node.sample(injected_deps,
                                                           key)))

            elif isinstance(node, distribution.ConditionedNode):
                model = graph.replace_node(
                    model, node.name, graph.ConstNode(node.name, node.value))

            else:
                raise TypeError("Unknown node type")

    return model
