# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

from typing import Callable, Dict

from piper.functional.modifier import Modifier
from piper import tree


class trace(Modifier):
    def __init__(self, fn: Callable):
        """Modifier to calculate log probability.

        Calculates the log probability of a stochastic function
        according to the log probabilities of individual
        distributions and sampled values.

        If you want to calculate the log probability of specific
        values under the model, condition on them appropriately.
        """
        super().__init__(fn)
        self.tree = tree.Tree()

    def __enter__(self):
        super().__enter__()
        return self

    def post_process(self, message: Dict):
        self.tree.add_node(
            message['name'],
            tree.Node(message['distribution'],
                      message['sample'],
                      message['is_conditioned']))

        return message

    def get_tree(self):
        """Return traced tree.
        """
        return self.tree
