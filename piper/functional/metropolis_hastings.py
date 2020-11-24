# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

from typing import List, Dict

from piper.models import forward
from piper.models import metropolis_hastings as mh
from piper.models import proposal as p


def proposal(model: forward.ForwardModel,
             condition_node: str) -> p.ProposalModel:
    """Initializes a proposal model for Metropolis-Hastings sampling.

    Args:
        model: ForwardModel to be sampled from.
        condition_node: Node to be used as the condition of the
            current value, i.e. x in P(x'|x). Needs to be an instance
            of ConstNode.

    Returns:
        An instance of ProposalModel.
    """
    return p.ProposalModel(model, condition_node)


def metropolis_hastings(model: forward.ForwardModel,
                        proposal: p.ProposalModel,
                        initial_samples: List[Dict],
                        burnin_steps: int = 100,
                        num_chains: int = 10) -> mh.MetropolisHastingsModel:
    """Initializes an MetropolisHastings model from a forward model.

    Args:
        model: ForwardModel to be sampled from.
        proposal: Proposal distribution. This should output samples for all
            chains.
        initial_samples: Dictionary of initial parameters for all chains.
            Should contain all DistributionNodes in the model.
        burnin_steps: Number of burn-in steps.
        num_chains: Number of chains to run in parallel.

    Returns:
        MetropolisHastings model.
    """
    return mh.MetropolisHastingsModel(model, proposal, initial_samples,
                                      burnin_steps, num_chains)
