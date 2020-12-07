#!/usr/bin/env python
from setuptools import setup

__version__ = '0.1'

with open("README.md", "r") as fh:
        long_description = fh.read()

setup(name = 'gqp_mc',
      version = __version__,
      author='ChangHoon Hahn', 
      author_email='hahn.changhoon@gmail.com', 
      description='python package for the DESI GQP WG mock challenge', 
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/changhoonhahn/gqp_mc/", 
      #packages=setuptools.find_packages(),
      install_requires = ['numpy', 'matplotlib', 'scipy', 'h5py', 'astropy', 'emcee', 'speclite', 'configobj', 'sqlalchemy', 'corner', 'multiprocess', 'pytest'],
      provides = ['gqp_mc'],
      include_package_data=True, 
      packages = ['gqp_mc', 'gqp_mc.firefly'],
      package_data={'gpc_mc': ['dat/*.pkl', 'dat/*.txt']},
      python_requires='>3.6'
      )
