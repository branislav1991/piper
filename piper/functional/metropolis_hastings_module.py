# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

from typing import Dict, Callable

import jax
import jax.numpy as jnp

from piper.functional.log_prob_module import log_prob
from piper.functional.condition_module import condition


class metropolis_hastings:
    def __init__(self,
                 model: Callable,
                 proposal: Callable,
                 initial_samples: Dict,
                 num_chains: int = 10,
                 proposal_seed: int = 1000):
        """Initializes an MetropolisHastings model from a forward model.

        Args:
            model: Model to be sampled from.
            proposal: Proposal distribution. Has to take in EXACTLY
                these parameters as input:
                    - the random key.
                    - keyword arguments with current sample values
                It needs to return a dictionary with proposed sample values.
            initial_samples: Dictionary of initial parameters.
                Has to contain all unconditioned distributions in the model.
            num_chains: Number of chains to run in parallel.
            proposal_seed: Random seed to use for generating proposals.

        Returns:
            Callable model.
        """
        self.model = model
        self.current_samples = {
            k: jnp.repeat(v[None, ...], num_chains, axis=0)
            for k, v in initial_samples.items()
        }
        self.proposal = proposal
        self.num_chains = num_chains
        self.key = jax.random.PRNGKey(proposal_seed)

        def parallel(key, current, args, kwargs):
            proposed_sample = self.proposal(key, **current)

            new_loglik_proposal = log_prob(
                condition(self.proposal, proposed_sample))
            new_loglik_proposal(key, **current)
            new_loglik_proposal = new_loglik_proposal.get()

            old_loglik_proposal = log_prob(condition(self.proposal, current))
            old_loglik_proposal(key, **proposed_sample)
            old_loglik_proposal = old_loglik_proposal.get()

            new_loglik_model = log_prob(condition(self.model, proposed_sample))
            new_loglik_model(*args, **kwargs)
            new_loglik_model = new_loglik_model.get()

            old_loglik_model = log_prob(condition(self.model, current))
            old_loglik_model(*args, **kwargs)
            old_loglik_model = old_loglik_model.get()

            new_loglik = new_loglik_model + new_loglik_proposal
            old_loglik = old_loglik_model + old_loglik_proposal

            u = jax.random.uniform(key,
                                   shape=next(iter(
                                       proposed_sample.values())).shape)

            def accept_reject(p, c):
                return jnp.where(u < jnp.exp(new_loglik - old_loglik), p, c)

            res = {}
            for name in proposed_sample:
                res[name] = accept_reject(proposed_sample[name], current[name])
            return res

        self.parallel = jax.jit(parallel)

    def __call__(self, *args, **kwargs):
        """Samples from the model.
        Args:
            key: JAX PRNG key.
        Returns:
            Dictionary of sampled random variables.
        """
        proposal_keys = jax.random.split(self.key, 1 + self.num_chains)
        self.key, proposal_keys = proposal_keys[0], proposal_keys[1:]

        res = jax.vmap(self.parallel,
                       in_axes=(0, 0, None, None))(proposal_keys,
                                                   self.current_samples, args,
                                                   kwargs)

        self.current_samples = res
        return res
