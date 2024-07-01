from setuptools import setup, find_packages

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sklearnlite",
    version="0.1.0",
    author="Ishi",
    author_email="shilpa.musale02@gmail.com",
    description="A tiny sklearn library with some of the machine learning models implemented",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ishi3012/sklearnlite",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
)