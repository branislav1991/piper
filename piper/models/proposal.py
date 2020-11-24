# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

from typing import Dict

import jax.numpy as jnp

from piper import core
from piper.models import forward


class ProposalModel(forward.ForwardModel):
    def __init__(self, model: forward.ForwardModel, condition_nodes: Dict):
        """Initializes a proposal model for Metropolis-Hastings sampling.

        Args:
            model: ForwardModel to be sampled from.
            condition_nodes: Dict of nodes to condition. The format is:
                {'n1': 'n3'}, where 'n1' is the ConstNode that will be
                conditioned, i.e. x in P(x'|x) and 'n3' is the node x'
                in the expression.
        Returns:
            An instance of ProposalModel.
        """
        super().__init__()

        if not model.can_sample():
            raise ValueError('Supplied ForwardModel cannot sample')

        for constn, samplen in condition_nodes.items():
            if constn not in model or not isinstance(model[constn],
                                                     core.ConstNode):
                raise ValueError('condition_node needs to be \
                        in the model and be a ConstNode')
            if samplen not in model:
                raise ValueError('sample_node needs to be \
                        in the model')

        self.nodes = model.nodes
        self.condition_nodes = condition_nodes
        self.old_values = {}

    def add(self, node: core.Node):
        raise NotImplementedError(
            "Please add nodes to ForwardModel and then apply \
                functional.proposal")

    def can_sample(self) -> bool:
        """Checks if you can propose new samples using the model.

        Returns:
            True.
        """
        return True

    def _save_const_vals(self) -> None:
        """Function to store current values of ConstNodes.

        Useful for proposals and log_prob computation since values need to be
        changed temporarily for these functions.
        """
        self.old_values = {}
        for constn in self.condition_nodes:
            self.old_values[constn] = self.nodes[constn].value

    def _restore_const_vals(self) -> None:
        """Function to restore old values of ConstNodes saved by _save_const_vals.
        """
        if not self.old_values:
            return

        for constn in self.old_values:
            self.nodes[constn].value = self.old_values[constn]

        self.old_values = {}

    def propose(self, current_values: Dict, key: jnp.ndarray) -> Dict:
        """Proposes new samples based on current values of samples.
        """
        self._save_const_vals()

        for constn, samplen in self.condition_nodes.items():
            self.nodes[constn].value = current_values[samplen]

        new_sample = self.sample(key)

        # remove condition_nodes from new_sample as they are not needed
        for constn in self.condition_nodes:
            del new_sample[constn]

        self._restore_const_vals()

        return new_sample

    def log_prob_proposal(self, values: Dict,
                          current_values: Dict) -> jnp.array:
        """Computes log probability using values and current_values.
        """
        self._save_const_vals()

        for constn, samplen in self.condition_nodes.items():
            self.nodes[constn].value = current_values[samplen]

        log_prob_val = self.log_prob(values)

        self._restore_const_vals()

        return log_prob_val
