# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

from typing import Optional

from piper import graph
from piper.distributions import distribution


def sample(model: graph.Graph, seed: Optional[int] = None) -> dict:
    layers = model.topological_sort()
    inferred = {}
    for layer in layers:
        for node in layer:
            if isinstance(node, distribution.Distribution):
                inferred[node.name] = node.sample(seed)
            else:
                raise TypeError("Unknown node type")

    return inferred
