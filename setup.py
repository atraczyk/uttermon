"""Setup file"""
try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

setup(
    name = 'uttermon',
    packages = find_packages(),
)
