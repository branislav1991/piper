# Copyright (c) 2020 Branislav Holländer. All rights reserved.
# See the file LICENSE for copying permission.

_KL_REGISTRY = {}
_MODIFIER_STACK = []


def register_kl(dist1, dist2):
    def decorator(fn):
        _KL_REGISTRY[(dist1, dist2)] = fn
        return fn

    return decorator
