# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Set up Environment."""

import timesformer.utils.logging as logging

_ENV_SETUP_DONE = False


def setup_environment():
    global _ENV_SETUP_DONE
    if _ENV_SETUP_DONE:
        return
    _ENV_SETUP_DONE = True
