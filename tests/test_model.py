# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

import pytest

import piper
from piper import core


class MockNode(core.Node):
    # Mock Node instantiation
    def __init__(self, name, dependencies=[]):
        super().__init__(name)
        self.dependencies = dependencies


def test_add_node():
    g = piper.create_forward_model()

    n1 = MockNode("n1")
    n2 = MockNode("n1")

    with pytest.raises(ValueError):
        g.add(n1)
        g.add(n2)

    n3 = MockNode("n3")

    with pytest.raises(ValueError):
        g.add(n3)
        g.add(n3)


def test_topological_sort():
    g = piper.create_forward_model()
    n1 = MockNode("n1")
    g.add(n1)
    n2 = MockNode("n2", ["n1"])
    g.add(n2)
    n3 = MockNode("n3", ["n1"])
    g.add(n3)
    n4 = MockNode("n4", ["n3"])
    g.add(n4)

    layers = g.topological_sort()
    assert layers
    assert len(layers) == 3
    assert len(layers[0]) == 1
    assert len(layers[1]) == 2
    assert len(layers[2]) == 1

    assert layers[0][0] == n1
    assert set(layers[1]) == set([n2, n3])
    assert layers[2][0] == n4
