# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

import abc
from abc import abstractmethod
import collections

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


class ConditionedNode(Node):
    def __init__(self, node: Node, value: jnp.ndarray):
        super().__init__(node.name)
        if not isinstance(node, DistributionNode):
            raise ValueError("Conditioned node must be a DistributionNode")

        self.value = value
        self.dependencies = node.dependencies


def conditioned_node(node: Node, value: jnp.ndarray):
    return ConditionedNode(node, value)


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

    def topological_sort(self) -> list:
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


class SampledModel(Model):
    def can_sample(self) -> bool:
        return False  # cannot sample if already sampled


def create_sampled_model() -> SampledModel:
    return SampledModel()


class ForwardModel(Model):
    def can_sample(self) -> bool:
        """Checks if you can apply forward sampling to the model.

        Returns:
            True if there is no conditioning on downstream nodes which have
            non-conditioned and non-const dependencies (otherwise you have
            to estimate posterior).
            False otherwise.
        """
        visited = set()
        queue = collections.deque()
        for node in self.nodes.values():
            if isinstance(node, ConditionedNode):
                queue.append(node)

        while queue:
            node = queue.popleft()
            if node in visited:
                continue

            visited.add(node)
            if (not isinstance(node, ConditionedNode)) \
                    and (not isinstance(node, ConstNode)):
                return False

            for d in node.dependencies:
                queue.append(self.nodes[d])

        return True


def create_forward_model() -> ForwardModel:
    return ForwardModel()


def replace_node(model: Model, node_name: str, new_node: Node) -> Model:
    """Replaces a node in the graph by a new node.

    Returns:
        Model with replaced node.
    """
    if node_name not in model:
        raise ValueError(f'Node {node_name} not in model')

    model.nodes[node_name] = new_node
    return model
