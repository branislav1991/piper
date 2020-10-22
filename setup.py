#!/usr/bin/python

# Copyright (c) 2020 Branislav Holländer

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import pathlib
from setuptools import find_packages, setup

__version__ = None

with open('piper/version.py') as f:
    exec(f.read(), globals())

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = ('README.md').read_text()

setup(
    name='piper',
    version=__version__,
    license='MIT',
    author='Branislav Holländer',
    author_email='branislav.hollander@gmail.com',
    description='Probabilistic Programming Using JAX',
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/branislav1991/piper",
    packages=find_packages(),
    install_requires=["numpy", "jax", "jaxlib"],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering',
    ],
    python_requires='>=3.7',
)
