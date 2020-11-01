# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

import pytest

import jax.numpy as jnp

from piper import param


def test_create_const_param():
    p = param.const_param(jnp.array([[1, 10, 5], [2, 3, 1]]))
    assert isinstance(p, param.ConstParam)


def test_create_dependent_param():
    p = param.dependent_param("hi")
    assert isinstance(p, param.DependentParam)


def test_create_flexible_param():
    with pytest.raises(ValueError):
        p = param.flexible_param(jnp.array([[1, 10, 5], [2, 3, 1]]))

    p = param.flexible_param(jnp.array(1))
    assert isinstance(p, param.FlexibleParam)


def test_convert_value_to_param():
    with pytest.raises(TypeError):
        param.to_param(10)

    p = param.to_param(jnp.array([[1, 10, 5], [2, 3, 1]]))
    assert isinstance(p, param.ConstParam)

    p = param.to_param("hi")
    assert isinstance(p, param.DependentParam)

    p = param.to_param(param.flexible_param(jnp.array(1)))
    assert isinstance(p, param.FlexibleParam)
