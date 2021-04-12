# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from setuptools import find_packages, setup

setup(
    name="timesformer",
    version="1.0",
    author="FBAI",
    url="unknown",
    description="TimeSformer",
    install_requires=[
        "yacs>=0.1.6",
        "pyyaml>=5.1",
        "av",
        "matplotlib",
        "termcolor>=1.1",
        "simplejson",
        "tqdm",
        "psutil",
        "matplotlib",
        "opencv-python",
        "pandas",
        "torchvision>=0.4.2",
        "sklearn",
        "tensorboard",
    ],
    extras_require={"tensorboard_video_visualization": ["moviepy"]},
    packages=find_packages(exclude=("configs", "tests")),
)
