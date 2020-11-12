# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

from piper.models import forward
import jax.numpy as jnp

from piper import core


class MCMCModel(core.Model):
    def __init__(self,
                 model: forward.ForwardModel,
                 proposal: forward.ForwardModel,
                 initial_params: dict,
                 burnin_steps: int):
        """Initializes the model from a ForwardModel.

        Args:
            model: ForwardModel to be sampled from.
            proposal: Proposal distribution.
            initial_params: Dictionary of initial parameters. Should contain
                all DistributionNodes in the model.
            burnin_steps: Number of burn-in steps.
        """
        super().__init__()

        self.nodes = model.nodes

        self.proposal = proposal
        self.burnin_steps = burnin_steps
        self.initial_params = initial_params
        self.ready_to_sample = False

        self.burnin()

    def burnin(self):
        self.ready_to_sample = True

    def add(self, node: core.Node):
        super().add(node)
        self.ready_to_sample = False

    def can_sample(self) -> bool:
        """Checks if you can apply MCMC sampling to the model.

        Returns:
            True.
        """
        return True

    def sample(self, key: jnp.ndarray) -> dict:
        """Samples from the model.

        Args:
            key: JAX PRNG key.

        Returns:
            Dictionary of sampled random variables.
        """
        if not self.ready_to_sample:
            raise RuntimeError(
                "Please perform burnin after creating the model or adding nodes"
            )

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
