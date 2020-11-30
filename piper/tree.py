# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

import jax.numpy as jnp

from piper.distributions.distribution import Distribution


class Node:
    def __init__(self,
                 dist: Distribution,
                 val: jnp.ndarray,
                 is_conditioned: bool):
        self.distribution = dist
        self.value = val
        self.is_conditioned = is_conditioned


class Tree:
    def __init__(self):
        self.nodes = {}

    def add_node(self, name: str, node: Node):
        if name in self.nodes:
            raise ValueError("Node already in the tree")

        self.nodes[name] = node
