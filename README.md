# ROMS-Tools
[![Conda version](https://img.shields.io/conda/vn/conda-forge/roms-tools.svg)](https://anaconda.org/conda-forge/roms-tools)
[![PyPI version](https://img.shields.io/pypi/v/roms-tools.svg)](https://pypi.org/project/roms-tools/)
[![codecov](https://codecov.io/gh/CWorthy-ocean/roms-tools/graph/badge.svg?token=5S1oNu39xE)](https://codecov.io/gh/CWorthy-ocean/roms-tools)
[![Documentation Status](https://readthedocs.org/projects/roms-tools/badge/?version=latest)](https://roms-tools.readthedocs.io/en/latest/?badge=latest)
![Run Tests](https://github.com/CWorthy-ocean/roms-tools/actions/workflows/tests.yaml/badge.svg)
![Supported Python Versions](https://img.shields.io/pypi/pyversions/roms-tools)

> [!Warning]
> **This project is still in an early phase of development.**
>
> The [Python API](https://roms-tools.readthedocs.io/en/latest/api.html) is not yet stable.
> Therefore whilst you are welcome to try out using the package, we cannot yet guarantee backwards compatibility.
> We expect to reach a more stable version in Q1 2025.

## Overview

A suite of Python tools for setting up and analyzing a [UCLA-ROMS](https://github.com/CESR-lab/ucla-roms) simulation.

## Installation

### ⚡️ **Installation from Conda-Forge**

To install `ROMS-Tools` with all dependencies, including `xesmf` and `dask`, use:

```bash
conda install -c conda-forge roms-tools
```

> [!Note]
>  Installation from Conda-Forge is the recommended installation method to ensure all features of `ROMS-Tools` are available.

### 📦 **Installation from PyPI (pip)**

You can also install `ROMS-Tools` from `pip`:

```bash
pip install roms-tools
```

If you want to use `ROMS-Tools` with `dask` (recommended for parallel and out-of-core computation), install it with the additional dependency:

```bash
pip install roms-tools[dask]
```

> [!Note]
>  The PyPI versions of `ROMS-Tools` do not include `xesmf`, so some features will be unavailable.


### Installation from GitHub

To obtain the latest development version, first clone the source repository:

```bash
git clone https://github.com/CWorthy-ocean/roms-tools.git
cd roms-tools
```

Next, install and activate the following conda environment:

```bash
conda env create -f ci/environment-with-xesmf.yml
conda activate romstools-test
```

Finally, install `ROMS-Tools` in the same environment:

```bash
pip install -e .
```

If you want to use `ROMS-Tools` with `dask` (recommended for parallel and out-of-core computation), you can
install `ROMS-Tools` along with the additional dependency via:

```bash
pip install -e ".[dask]"
```


### Run the tests

Before running the tests, you can activate the conda environment created in the previous section:

```bash
conda activate romstools-test
```

Check the installation of `ROMS-Tools` has worked by running the test suite
```bash
cd roms-tools
pytest
```

## Getting Started

To learn how to use `ROMS-Tools`, check out the [documentation](https://roms-tools.readthedocs.io/en/latest/).

## Feedback and contributions

**If you find a bug, have a feature suggestion, or any other kind of feedback, please start a Discussion.**

We also accept contributions in the form of Pull Requests.


## See also

- [ROMS source code](https://github.com/CESR-lab/ucla-roms)
- [C-Star](https://github.com/CWorthy-ocean/C-Star)
