# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

from piper.models import forward
from piper.models import mcmc as mcmcmodel


def mcmc(model: forward.ForwardModel,
         proposal: forward.ForwardModel,
         initial_params: dict,
         burnin_steps: int = 500) -> mcmcmodel.MCMCModel:
    """Initializes an MCMC model from a forward model.

    Args:
        model: ForwardModel to be sampled from.
        proposal: Proposal distribution.
        initial_params: Dictionary of initial parameters. Should contain
            all DistributionNodes in the model.
        burnin_steps: Number of burn-in steps.

    Returns:
        MCMC model.
    """
    return mcmcmodel.MCMCModel(model, proposal, initial_params, burnin_steps)
