# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

import collections

import jax.numpy as jnp

from piper import core


class ForwardModel(core.Model):
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
            if isinstance(node, core.ConditionedNode):
                queue.append(node)

        while queue:
            node = queue.popleft()
            if node in visited:
                continue

            visited.add(node)
            if (not isinstance(node, core.ConditionedNode)) \
                    and (not isinstance(node, core.ConstNode)):
                return False

            for d in node.dependencies:
                queue.append(self.nodes[d])

        return True

    def sample(self, key: jnp.ndarray) -> dict:
        """Samples from the model.

        Args:
            key: JAX PRNG key.

        Returns:
            Dictionary of sampled random variables.
        """
        if not self.can_sample():
            raise RuntimeError("Invalid forward sampling on \
                conditioned dependency")

        res = {}

        layers = self._topological_sort()
        for layer in layers:
            for node in layer:
                if isinstance(node, core.DistributionNode):
                    injected_deps = {}
                    for d in node.dependencies:
                        if d not in res:
                            raise RuntimeError("Invalid forward sampling on \
                                non-const dependency")

                        injected_deps[d] = res[d]

                    res[node.name] = node.sample(injected_deps, key)

                elif isinstance(node, core.ConditionedNode):
                    res[node.name] = node.value

                else:
                    raise TypeError("Unknown node type")

        return res


def create_forward_model() -> ForwardModel:
    return ForwardModel()
