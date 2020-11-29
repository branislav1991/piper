# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

from typing import Dict, Callable

import jax
import jax.numpy as jnp

from piper.functional.log_prob_module import log_prob
from piper.functional.trace_module import trace
from piper.functional.condition_module import condition
from piper import tree


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

            tracer = trace(condition(self.proposal, proposed_sample))
            tracer(key, **current)
            proposal_tree = tracer.get_tree()
            new_loglik_proposal = log_prob(proposal_tree)

            tracer = trace(condition(self.proposal, current))
            tracer(key, **proposed_sample)
            old_loglik_proposal = log_prob(tracer.get_tree())

            tracer = trace(condition(self.model, proposed_sample))
            tracer(*args, **kwargs)
            model_new = tracer.get_tree()
            new_loglik_model = log_prob(model_new)

            tracer = trace(condition(self.model, current))
            tracer(*args, **kwargs)
            model_old = tracer.get_tree()
            old_loglik_model = log_prob(model_old)

            new_loglik = new_loglik_model + new_loglik_proposal
            old_loglik = old_loglik_model + old_loglik_proposal

            if not self._check_all_conditioned(model_new, model_old):
                raise RuntimeError(
                    'Incompatible model, proposal and initial samples')

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

    def _check_all_conditioned(self,
                               model_new: tree.Tree,
                               model_old: tree.Tree):
        """Checks if all model variables are conditioned.

        proposal has to contain samples for all unconditioned variables
        in the model. current_samples has to contain the same variables as
        proposal.
        """
        for node in model_new.nodes.values():
            if not node.is_conditioned:
                return False

        for node in model_old.nodes.values():
            if not node.is_conditioned:
                return False

        return True

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
