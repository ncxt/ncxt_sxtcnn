from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools

__version__ = '0.0.1'


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)


ext_modules = [
    Extension(
        '_blocks',
        ['src/main.cpp'],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True)
        ],
        language='c++',
    ),
]

try:
    import torch
except ModuleNotFoundError:
    raise ModuleNotFoundError('''
    This module PyTorch 
    Check for suitable local installation at 
    https://pytorch.org/get-started/locally/
    ''')

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sxtcnn",
    version="0.0.1",
    author="Axel Ekman",
    author_email="axel.ekman@iki.fi",
    description="semantic segmentation for sxt data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ncxt/ncxt_sxtcnn",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy>=1.13.3', 'matplotlib>=3.0.3', 'scipy>=1.2.0', 'tqdm>=4.25.0'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
