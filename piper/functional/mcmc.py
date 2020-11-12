# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

from piper import models


def mcmc(model: models.forward.ForwardModel,
         proposal: models.forward.ForwardModel,
         initial_params: dict,
         burnin_steps: int = 500) -> models.mcmc.MCMCModel:
    """Initializes an MCMC model from a forward model.

    Args:
        model: Model to apply the conditioning to.
        node: Node to condition on.
        val: Value to condition with.

    Returns:
        New mode with ConditionedNode with the conditioned value.
    """
    return models.mcmc.MCMCModel(model, proposal, initial_params, burnin_steps)


def burnin(model: models.mcmc.MCMCModel) -> models.mcmc.MCMCModel:
    return model.burnin()
