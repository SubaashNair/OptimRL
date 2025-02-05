from setuptools import setup, find_packages
import os
import platform
import shutil
import versioneer
import sysconfig

# Read the README file
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="optimrl",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Subashanan Nair",
    author_email="subaashnair12@gmail.com",
    description="Group Relative Policy Optimization for Efficient RL Training",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SubaashNair/OptimRL",
    packages=find_packages(include=["optimrl", "optimrl.*"]),
    install_requires=[
        "numpy>=1.20.0",
        "torch>=1.8.0"
    ],
    extras_require={
        'test': ['pytest>=6.0', 'flake8', 'isort', 'black'],
        'dev': ['pytest>=6.0', 'black', 'isort', 'flake8']
    },
    python_requires=">=3.8",
    include_package_data=True
)