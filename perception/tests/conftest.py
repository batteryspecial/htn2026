"""Shared fixtures. Nothing here loads a real model: the detector tests that
need weights are marked `weights` and skipped unless PERCEPTION_TEST_WEIGHTS=1."""

import os

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "weights: needs real model weights on disk")


def pytest_collection_modifyitems(config, items):
    if os.environ.get("PERCEPTION_TEST_WEIGHTS") == "1":
        return
    skip = pytest.mark.skip(reason="set PERCEPTION_TEST_WEIGHTS=1 to run")
    for item in items:
        if "weights" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def frame():
    import numpy as np

    return np.zeros((480, 640, 3), dtype=np.uint8)
