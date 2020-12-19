# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

from piper import tree


def elbo(model_tree: tree.Tree, q_tree: tree.Tree):
    """Calculates evidence lower bound given the model tree
    and q tree.
    """
    elbo = 0.

    for name, node in model_tree.nodes.items():
        qval = q_tree.nodes[name]
        elbo += node.distribution.log_prob(qval.value)

    for node in q_tree.nodes.values():
        elbo -= node.distribution.log_prob(node.value)

    return -elbo
