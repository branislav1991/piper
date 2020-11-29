# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

from piper import tree


def log_prob(tr: tree.Tree):
    """Calculates log probability given the trace tree.

    If you want to calculate the log probability of specific
    values under the model, condition on them appropriately.
    """
    logp = 0

    for node in tr.nodes.values():
        logp += node.distribution.log_prob(node.value)

    return logp
