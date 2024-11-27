#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="slt_inference",
    version="0.0.1",
    author="Phillip Rust",
    author_email="philliprust@meta.com",
    url="https://github.com/fairinternal/slt_inference",
    description="Experimental code for sign language translation inference",
    license="",
    package_dir={"": "src"},
    packages=find_packages("src"),
    zip_safe=True,
)
