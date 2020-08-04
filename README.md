# Convolutional Conditional Neural Processes for Rainfall Runoff Modelling

This repository contains code for the "ConvCNPs for Hydrology" project, completed as part of the MRes in Environmental Data Science (AI4ER CDT) at the University of Cambridge. We use [Convolutional Convolutional Neural Processes](https://openreview.net/forum?id=Skey4eBYPS) for rainfall runoff modelling and other hydrological prediction tasks. 

* [Installation](#installation)
* [Expository Notebooks](#expository-notebooks)
* [Reproducing the 1D Experiments](#reproducing-the-1d-experiments)
* [Reference](#reference)

## Installation
Requirements:

* Python 3.6 or higher.

* `gcc` and `gfortran`:
    On OS X, these are both installed with `brew install gcc`.
    On Linux, `gcc` is most likely already available,
    and `gfortran` can be installed with `apt-get install gfortran`.
    
To begin with, clone and enter the repo.

```bash
git clone https://github.com/mgironamata/convcnp
cd convcnp
```

Then make a virtual environment and install the requirements.

```bash
virtualenv -p python3 venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

This will install the latest version of `torch`.
If your version of CUDA is not the latest version, then you might need to
install an earlier version of `torch`.

You should now be ready to go!
If you encounter any problems, feel free to open an issue, and will try to
help you resolve the problem as soon as possible.

