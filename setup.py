import os
import sys
from setuptools import setup, find_packages
here = os.path.abspath(os.path.dirname(__file__))

requires = ['numpy']

setup(name='DiffJPEG',
      version='1.0',
      description='DiffJPEG',
      author='mlomnitz',
      url='https://github.com/mlomnitz/DiffJPEG',
      keywords='JPEG',
      packages=find_packages(),
      license='LICENSE',
      install_requires=requires)
