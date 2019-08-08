#!/usr/bin/env python
from setuptools import setup

__version__ = '0.1'

setup(name = 'gqp_mc',
      version = __version__,
      python_requires='>3.5.2',
      description = 'GQP mock challenge',
      requires = ['numpy', 'matplotlib', 'scipy', 'h5py', 'astropy', 'emcee', 'fsps', 'speclite'],
      provides = ['gqp_mc'],
      packages = ['gqp_mc']
      )
