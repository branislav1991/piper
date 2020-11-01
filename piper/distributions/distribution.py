# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

import abc
import collections

import jax.numpy as jnp

from piper import graph
from piper import param


class DistributionNode(graph.Node):
    def __init__(self, name: str):
        super().__init__(name)

    @abc.abstractmethod
    def sample(self, dependencies: dict, key: jnp.ndarray):
        """Sample from the distribution.

        Args:
            dependencies: dict of dependencies.
            key: JAX random key.
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


def _get_samples(params: list, dependencies: dict) -> list:
    """Obtains samples from parameters of a node.

    Requires that all parameters have the same shape but
    does not check this requirement.

    Returns:
        List of samples in the order of params.
    """
    non_flex_samples = [
        p.get(dependencies) for p in params
        if not isinstance(p, param.FlexibleParam)
    ]

    if not non_flex_samples:
        raise ValueError("No unflexible params provided")

    shape = non_flex_samples[0].shape
    flex_param_samples = [
        p.get(dependencies, shape=shape) for p in params
        if isinstance(p, param.FlexibleParam)
    ]

    result = collections.deque()
    for i in range(len(params) - 1, -1, -1):
        if isinstance(params[i], param.FlexibleParam):
            result.appendleft(flex_param_samples.pop())
        else:
            result.appendleft(non_flex_samples.pop())

    return result


class ConditionedNode(graph.Node):
    def __init__(self, node: graph.Node, value: jnp.ndarray):
        super().__init__(node.name)
        if not isinstance(node, DistributionNode):
            raise ValueError("Conditioned node must be a DistributionNode")

        self.value = value
        self.dependencies = node.dependencies


def conditioned_node(node: graph.Node, value: jnp.ndarray):
    return ConditionedNode(node, value)
