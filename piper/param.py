# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

import abc
from typing import Union

import jax.numpy as jnp


class Param(abc.ABC):
    @abc.abstractmethod
    def get(self, dependencies: dict, **kwargs):
        raise NotImplementedError()


class ConstParam(Param):
    def __init__(self, value: jnp.ndarray):
        if not isinstance(value, jnp.ndarray):
            raise ValueError("ConstParam requires an ndarray as value")

        self.value = value

    def get(self, dependencies: dict, **kwargs):
        return self.value


def const_param(value: jnp.ndarray) -> ConstParam:
    return ConstParam(value)


class DependentParam(Param):
    def __init__(self, name: str):
        if not isinstance(name, str):
            raise ValueError(
                "DependentParam requires the parameter name as a str")

        self.name = name

    def get(self, dependencies: dict, **kwargs):
        if self.name not in dependencies:
            raise ValueError("Dependent value not provided")
        return dependencies[self.name]


def dependent_param(name: str) -> DependentParam:
    return DependentParam(name)


class FlexibleParam(Param):
    def __init__(self, value: jnp.ndarray):
        if not isinstance(value, jnp.ndarray) or value.shape != ():
            raise ValueError('val needs to be of type ndarray and of \
                empty shape')

        self.value = value

    def get(self, dependencies: dict, **kwargs):
        if 'shape' not in kwargs:
            raise ValueError("shape not provided")

        shape = kwargs['shape']
        return self.value.repeat(shape)


def flexible_param(value: jnp.ndarray):
    return FlexibleParam(value)


def to_param(val: Union[jnp.ndarray, str, Param]):
    """Convert ndarray or str to an instance of Param.

    If a Param is provided, leave it untouched.

    Returns:
        Param based on the input.
    """
    if isinstance(val, jnp.ndarray):
        return const_param(val)
    elif isinstance(val, str):
        return dependent_param(val)
    elif isinstance(val, Param):
        return val
    else:
        raise TypeError("val cannot be converted to a Param")
