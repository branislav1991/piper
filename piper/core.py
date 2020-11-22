# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

import abc
from abc import abstractmethod
import collections
import copy
from typing import Dict, List

import jax.numpy as jnp

from piper import param


class Node(abc.ABC):
    def __init__(self, name: str):
        self.name = name
        self.dependencies = []


class ConstNode(Node):
    def __init__(self, name: str, value: jnp.ndarray):
        super().__init__(name)

        self.value = value


def const_node(name: str, value: jnp.ndarray):
    return ConstNode(name, value)


class DistributionNode(Node):
    """Parent class for all distributions.

    Allows one to condition on the distribution.
    """
    def __init__(self, name: str):
        super().__init__(name)
        self._condition = None

    def is_conditioned(self) -> bool:
        return self._condition is not None

    def condition(self, val: jnp.ndarray):
        self._condition = val

    def sample(self, dependencies: Dict, key: jnp.ndarray):
        """Sample from the distribution.

        Args:
            dependencies: Dict of dependencies.
            key: JAX random key.
        """
        if self.is_conditioned():
            return self._condition

        return self._sample(dependencies, key)

    @abc.abstractmethod
    def _sample(self, dependencies: Dict, key: jnp.ndarray):
        raise NotImplementedError

    def log_prob(self, values: Dict) -> jnp.ndarray:
        """Returns log probability density for a value.

        Accepts a single Dict of values which includes the value
        for this distribution (if it is not conditioned)
        as well as for any dependencies.

        Returns:
            Log probability of x under the distribution.
        """
        values_ = copy.copy(values)
        if self.is_conditioned():
            values_[self.name] = self._condition

        return self._log_prob(values_[self.name], values_)

    @abc.abstractmethod
    def _log_prob(self, x: jnp.ndarray, dependencies: Dict) -> jnp.ndarray:
        """Returns log probability density for a value.

        Args:
            x: Values to be evaluated. Either a single value or a
                batch of values along dimension 0.

        Returns:
            Log probability of x under the distribution.
        """
        raise NotImplementedError

    def _get_samples(self, params: List, dependencies: Dict) -> List:
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


class Model(abc.ABC):
    """Abstract base class for all models.

    Implements some basic functionality such as adding nodes
    to the graph or topological sorting.
    """
    def __init__(self):
        self.nodes = {}  # Nodes by name

    @abstractmethod
    def can_sample(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def sample(self) -> Dict:
        raise NotImplementedError()

    def log_prob(self, values: Dict) -> jnp.array:
        """Calculates log probability of entire model given values for all nodes.

        Args:
            values: Values for log probability calculation. Values for all
                DistributionNodes need to be provided.

        Returns:
            Log probability of model.
        """
        logp = 0
        for name, node in self.nodes.items():
            if isinstance(node, DistributionNode):
                logp += node.log_prob(values)

        return logp

    def add(self, node: Node):
        if node.name in self.nodes:
            raise ValueError(
                f"Cannot register multiple instances of '{node.name}' \
                             in the graph")

        self.nodes[node.name] = node

        for dep in node.dependencies:
            if dep not in self.nodes:
                raise ValueError(f"Cannot find dependency {dep} in the graph")

    def __contains__(self, nodename: str):
        return nodename in self.nodes

    def __getitem__(self, key):
        return self.nodes[key]

    def _topological_sort(self) -> List:
        """Return nodes topologically sorted based on their dependencies.

        Returns:
            List of lists of nodes sorted according the dependencies.
        """
        if not self.nodes:
            return []

        layers = []
        visited = set()
        while len(visited) < len(self.nodes):
            new_layer = []
            new_names = set()
            for name, node in self.nodes.items():
                if name in visited:
                    continue

                resolved = True
                for d in node.dependencies:
                    if d not in visited:
                        resolved = False
                        break

                if resolved:
                    new_layer.append(node)
                    new_names.add(name)

            visited = visited.union(new_names)
            layers.append(new_layer)

        return layers
