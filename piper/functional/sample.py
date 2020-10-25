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
                injected_deps = {}
                for d in node.dependencies:
                    injected_deps[d] = inferred[d]

                inferred[node.name] = node.sample(seed, **injected_deps)
            else:
                raise TypeError("Unknown node type")

    return inferred
