# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

import jax
import jax.numpy as jnp

import piper.functional as func
from piper import distributions as dist
from piper import test_util as tu


def test_sample_dirichlet():
    def model1(key):
        return func.sample('n', dist.dirichlet(jnp.array([0.5, 0.5])), key)

    keys = jax.random.split(jax.random.PRNGKey(123), 100)
    samples = jax.vmap(lambda k: model1(k))(keys)

    tu.check_close(jnp.mean(samples, 0), jnp.array([0.5196701, 0.48032987]))

    def model2(key):
        return func.sample('n', dist.dirichlet(jnp.array([2., 3., 0.5])), key)

    samples = jax.vmap(lambda k: model2(k))(keys)

    tu.check_close(jnp.mean(samples, 0),
                   jnp.array([0.37348172, 0.5442406, 0.08227774]))

    def model3(key):
        return func.sample('n', dist.dirichlet(jnp.full((2, 3), 0.5)), key)

    keys = jax.random.split(jax.random.PRNGKey(123), 100)
    samples = jax.vmap(lambda k: model3(k))(keys)

    tu.check_close(
        jnp.mean(samples, 0),
        jnp.array([[0.30348125, 0.3329056, 0.36361322],
                   [0.37877005, 0.3130015, 0.3082284]]))


def test_kl_dir_dir_one_dimensional():
    n1 = dist.dirichlet(jnp.array([1.0, 0.5, 0.5]))
    n2 = dist.dirichlet(jnp.array([1.0, 0.5, 0.5]))
    n3 = dist.dirichlet(jnp.array([0.5, 2.0, 2.5]))

    tu.check_close(func.compute_kl_div(n1, n2), 0.)
    tu.check_close(func.compute_kl_div(n1, n3), 4.386292)


def test_kl_dir_dir_multi_dimensional():
    n1 = dist.dirichlet(jnp.array([[1.0, 0.5, 0.5], [1.0, 0.5, 0.5]]))
    n2 = dist.dirichlet(jnp.array([[1.0, 0.5, 0.5], [0.5, 0.5, 1.0]]))

    tu.check_close(func.compute_kl_div(n1, n2), jnp.array([0, 0.69314706]))


def test_sample_dir_invalid_value_error():
    def model(key):
        n1 = func.sample('n1', dist.dirichlet(jnp.array([-1., 1.])), key)
        return n1

    key = jax.random.PRNGKey(123)
    sample = model(key)
    assert jnp.all(jnp.isnan(sample))


def test_sample_dir_conditioned_invalid_value_error():
    def model(key):
        n1 = func.sample('n1', dist.dirichlet(jnp.array([1.0, 0.5])), key)
        return n1

    conditioned_model = func.condition(model, {'n1': jnp.array([0.8, 0.7])})
    key = jax.random.PRNGKey(123)
    sample = conditioned_model(key)
    assert jnp.all(jnp.isnan(sample))


def test_log_prob_dir():
    n1 = dist.dirichlet(jnp.array([0.4, 5.0, 15.0]))
    log_prob_1 = n1.log_prob(jnp.array([0.2, 0.2, 0.6]))
    log_prob_2 = n1.log_prob(jnp.array([0.1, 0.3, 0.6]))
    tu.check_close(log_prob_1, -1.257432)
    tu.check_close(log_prob_2, 0.7803173)
