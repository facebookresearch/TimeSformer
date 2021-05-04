# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from setuptools import find_packages, setup

setup(
    name="timesformer",
    version="1.0",
    author="FBAI",
    url="unknown",
    description="TimeSformer",
    keywords = [
    'artificial intelligence',
    'attention mechanism',
    'transformers',
    'video classification',
    ],
    install_requires=[
        'einops>=0.3',
        'torch>=1.6'
    ],
    extras_require={"tensorboard_video_visualization": ["moviepy"]},
    packages=find_packages(exclude=("configs", "tests")),
)
