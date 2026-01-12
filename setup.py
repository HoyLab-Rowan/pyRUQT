#!/usr/bin/env python3

import os
import re
import sys
from setuptools import setup,find_packages

python_min_version = (3, 9)
python_requires = '>=' + '.'.join(str(num) for num in python_min_version)

if sys.version_info < python_min_version:
    raise SystemExit('Python 3.9 or later is required!')


install_requires = [
    'numpy>=1.21.6',  
    'scipy>=1.8.1',  
    'matplotlib>=3.5.2',  
    'importlib-metadata>=0.12;python_version<"3.8"'
]

setup(name="pyRUQT",
      version="1.0.0",
      description="Rowan University Quantum Transport (Python-version)",
      url="https://github.com/HoyLab-Rowan",
      maintainer="Hoy Research Group",
      author='Erik P. Hoy',
      author_email="hoy@rowan.edu", 
      license='GPL-3.0',
      platforms=['unix'],
      py_modules=['pyruqt','ruqt'],
      packages=find_package(),
      python_requires='>=' + '.'.join(str(num) for num in python_min_version),
      install_requires=install_requires,
      classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Lesser General Public License v3 or later (GPLv3.0)',  
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',        
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics'
      ],
)
