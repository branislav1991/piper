# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

import jax
import jax.numpy as jnp

from piper import core
from piper.models import forward


class MCMCModel(core.Model):
    def __init__(self, model: forward.ForwardModel,
                 proposal: forward.ForwardModel, initial_samples: dict,
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
        self.initial_samples = initial_samples
        self.current_samples = self.initial_samples

        self.burnin()

    def burnin(self):
        keys = jax.random.split(jax.random.PRNGKey(123), self.burnin_steps)
        for i in range(self.burnin_steps):
            self.sample(keys[i])

    def add(self, node: core.Node):
        raise NotImplementedError(
            "Please add nodes to ForwardModel and then apply functional.mcmc")

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
        res = {}

        proposed_samples = self.proposal.sample(key)
        new_loglik = self.logp(proposed_samples) + self.proposal.logp(
            proposed_samples)
        old_loglik = self.logp(self.current_samples) + self.proposal.logp(
            self.current_samples)

        if new_loglik > old_loglik:
            res = proposed_samples
        else:
            u = jax.random.uniform(key)
            if u < jnp.exp(new_loglik - old_loglik):
                res = proposed_samples
            else:
                res = self.current_samples

        self.current_samples = res
        return res
