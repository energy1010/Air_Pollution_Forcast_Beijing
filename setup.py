#!/usr/bin/env python
# encoding:utf-8

from setuptools import setup, find_packages

setup(
    name = "Air_Pollution_Forcast_Beijing",
    version = "1.0",
    keywords = ["test", "xxx"],
    description = "eds sdk",
    long_description = "eds sdk for python",
    license = "MIT Licence",

    url = "http://test.com",
    author = "energy1010",
    author_email = "test@gmail.com",

    packages = find_packages(),
    include_package_data = True,
    platforms = "any",
    install_requires = [],

    scripts = [],
    entry_points = {
        'console_scripts': [
        #'test = test.help:main'
        ]
    }
)
