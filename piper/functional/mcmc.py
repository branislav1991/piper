# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

from typing import List, Dict

from piper.models import forward
from piper.models import mcmc as mcmcmodel


def mcmc(model: forward.ForwardModel,
         proposal: forward.ForwardModel,
         initial_samples: List[Dict],
         burnin_steps: int = 500,
         num_chains: int = 10) -> mcmcmodel.MCMCModel:
    """Initializes an MCMC model from a forward model.

    Args:
        model: ForwardModel to be sampled from.
        proposal: Proposal distribution. This should output samples for all chains.
        initial_samples: Dictionary of initial parameters for all chains.
            Should contain all DistributionNodes in the model.
        burnin_steps: Number of burn-in steps.
        num_chains: Number of chains to run in parallel.

    Returns:
        MCMC model.
    """
    return mcmcmodel.MCMCModel(model, proposal, initial_samples, burnin_steps, num_chains)
