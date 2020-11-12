# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

import pytest

import piper
from piper import core
import piper.models as models


def test_mcmc():
    g = models.create_forward_model()

    n1 = MockNode("n1")
    n2 = MockNode("n1")

    with pytest.raises(ValueError):
        g.add(n1)
        g.add(n2)

    n3 = MockNode("n3")

    with pytest.raises(ValueError):
        g.add(n3)
        g.add(n3)
