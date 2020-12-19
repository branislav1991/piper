# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

from typing import Callable, Dict

import jax
import jax.experimental.optimizers as jax_optim

from piper.functional.trace_module import trace


class svi:
    def __init__(self, model: Callable, q: Callable, loss: Callable,
                 optimizer: jax_optim.Optimizer, initial_params: Dict):
        """Initializes stochastic variational inference.

        Args:
            model: model to be approximated.
            q: Approximation model. Needs to have the same inputs, outputs
                and distributions as model, as well as arbitrary variational
                parameters as a dictionary. The parameter holding the dictionary
                of variational parameters must be the
                first parameter of the function.
            loss: Function to calculate the loss given
                the traces of model and q.
            optimizer: JAX optimizer.
            initial_params: Initial params for optimization.
                Has to correspond to the input params in q.
        """
        self.model = model
        self.q = q
        self.loss = loss
        self.optimizer = optimizer
        self.optimizer_state = self.optimizer.init_fn(initial_params)
        self.step = 0

    def __call__(self, *args, **kwargs):
        def loss(params, *args, **kwargs):
            tracer_model = trace(self.model)
            tracer_model(*args, **kwargs)
            model_tree = tracer_model.get_tree()

            tracer_q = trace(self.q)
            tracer_q(params, *args, **kwargs)
            q_tree = tracer_q.get_tree()

            return self.loss(model_tree, q_tree)

        gradient_fn = jax.value_and_grad(loss)
        value, gradients = gradient_fn(
            self.optimizer.params_fn(self.optimizer_state), *args, **kwargs)
        self.optimizer_state = self.optimizer.update_fn(
            self.step,
            gradients,
            self.optimizer_state)
        self.step += 1
        return value

    def get_param(self, param_name):
        return self.optimizer.params_fn(self.optimizer_state)[param_name]
