#!/usr/bin/env python3

import os
import re
import sys
from setuptools import setup,find_packages
from setuptools import setuptools.command,build_py import build_py as _build_py

#Check that python packages needed by ASE (as of August 2022)  are in place.
#Modified from ASE setup.py file available here: https://wiki.fysik.dtu.dk/ase
python_min_version = (3, 6)
python_requires = '>=' + '.'.join(str(num) for num in python_min_version)


if sys.version_info < python_min_version:
    raise SystemExit('Python 3.6 or later is required!')


install_requires = [
    'numpy>=1.17.0',  
    'scipy>=1.3.1',  
    'matplotlib>=3.1.0',  
    'importlib-metadata>=0.12;python_version<"3.8"'
]

setup(name="pyRUQT",
      version="0.1"
      description="Rowan University Quantum Transport (Python-version)",
      url="https://github.com/HoyLab-Rowan",
      maintainer="Hoy Research Group",
      author='Erik P. Hoy',
      author_email="hoy@rowan.edu", 
      license='GPL-3.0',
      platforms=['unix'],
      packages=find_package(),
      python_requires=python_pythongs,
      install_requires=install_requires,
      classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Lesser General Public License v3 or later (GPLv3.0)',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Physics'
      ],
)
