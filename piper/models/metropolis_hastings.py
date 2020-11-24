# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

from typing import Dict

import jax
import jax.numpy as jnp

from piper import core
from piper.models import forward
from piper.models import proposal


class MetropolisHastingsModel(core.Model):
    def __init__(self, model: forward.ForwardModel,
                 proposal: proposal.ProposalModel, initial_samples: Dict,
                 burnin_steps: int, num_chains: int):
        """Initializes a MetropolisHastings model from a ForwardModel.

        This model allows you to use the Metropolis-Hastings algorithm to sample
        from a distribution conditioned on arbitrary nodes.
        It is not modifiable and you cannot add new nodes to it.

        Args:
            model: ForwardModel to be sampled from.
            proposal: Proposal distribution. Needs to be an instance of
                ProposalModel. Will output proposals according to P(x'|x).
                Should output samples for all chains.
            initial_samples: Dictionary of initial parameters for all chains.
                Should contain all DistributionNodes in the model.
            burnin_steps: Number of burn-in steps.
            num_chains: Number of chains to run in parallel.
        """
        super().__init__()

        def metropolis_hastings(u, new_loglik, old_loglik, proposed_samples,
                                current_samples):
            new_val = jnp.where(u < jnp.exp(new_loglik - old_loglik),
                                proposed_samples, current_samples)

            return new_val

        self.metropolis_hastings = jax.jit(metropolis_hastings)

        self.nodes = model.nodes

        self.proposal = proposal
        self.burnin_steps = burnin_steps
        self.initial_samples = initial_samples
        self.current_samples = self.initial_samples
        self.num_chains = num_chains

        self.burnin()

    def burnin(self):
        keys = jax.random.split(jax.random.PRNGKey(102030), self.burnin_steps)
        for i in range(self.burnin_steps):
            self.sample(keys[i])

    def add(self, node: core.Node):
        raise NotImplementedError(
            "Please add nodes to ForwardModel and then apply \
                functional.metropolis_hastings")

    def can_sample(self) -> bool:
        """Checks if you can apply MetropolisHastings sampling to the model.

        Returns:
            True.
        """
        return True

    def sample(self, key: jnp.ndarray) -> Dict:
        """Samples from the model.

        Args:
            key: JAX PRNG key.

        Returns:
            Dictionary of sampled random variables.
        """
        proposed_samples = self.proposal.propose(self.current_samples, key)
        res = {}

        new_loglik = self.log_prob(
            proposed_samples) + self.proposal.log_prob_proposal(
                self.current_samples, proposed_samples)
        old_loglik = self.log_prob(
            self.current_samples) + self.proposal.log_prob_proposal(
                proposed_samples, self.current_samples)

        u = jax.random.uniform(key,
                               shape=next(iter(
                                   proposed_samples.values())).shape)
        if u.shape[0] != self.num_chains:
            raise ValueError(
                "First dim of proposed samples needs to be num_chains")

        for name, val in proposed_samples.items():
            res[name] = self.metropolis_hastings(u, new_loglik, old_loglik,
                                                 val,
                                                 self.current_samples[name])

        self.current_samples = res
        return res