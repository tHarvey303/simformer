from setuptools import Command, find_packages, setup
import os

NAME = "scoresbibm"
VERSION = "0.0.1"
DESCRIPTION = "Score-based inference benchmark"
AUTHOR = "Anonymous"

entry_points={
    'console_scripts': [
        'scoresbi = scoresbibm.scripts:main',
    ],
}



REQUIRED = [
    "numpy",
    "matplotlib",
    "jax",
    "torch==2.1.0",
    "torchaudio",
    "torchvision",
    "hydra-core",
    "hydra-submitit-launcher",
    "hydra-optuna-sweeper",
    "omegaconf",
    "sbi==0.22.0",
    "optuna",
    "tueplots",
    "seaborn",
    "pandas",
]

setup(name=NAME,version=VERSION, description=DESCRIPTION, author=AUTHOR,package_dir={"scoresbibm": "scoresbibm"}, install_requires=REQUIRED, entry_points=entry_points)

# Do not print the output of the following command
os.system("pip install sbibm --no-deps -q")