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
        model: Model to apply the conditioning to.
        node: Node to condition on.
        val: Value to condition with.

    Returns:
        New mode with ConditionedNode with the conditioned value.
    """
    return mcmcmodel.MCMCModel(model, proposal, initial_params, burnin_steps)
