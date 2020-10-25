# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

from piper import graph

kl_registry = {}


def register_kl(dist1, dist2):
    def decorator(fn):
        kl_registry[(dist1, dist2)] = fn
        return fn

    return decorator


def kl_divergence(model: graph.Graph, dist1: str, dist2: str):
    if dist1 not in model or dist2 not in model:
        raise ValueError(f"{dist1} or {dist2} not defined in model")

    node1 = model.nodes[dist1]
    node2 = model.nodes[dist2]

    if (type(node1), type(node2)) not in kl_registry:
        raise ValueError(
            f"KL-divergence between {type(node1)} and {type(node2)} \
             not defined")

    fn = kl_registry[(type(node1), type(node2))]
    return fn(node1, node2)
