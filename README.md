# hydrodiy
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10065114.svg)](https://doi.org/10.5281/zenodo.10065114) 
[![CI](https://github.com/csiro-hydroinformatics/hydrodiy/actions/workflows/python-package-conda.yml/badge.svg)](https://github.com/csiro-hydroinformatics/hydrodiy/actions/workflows/python-package-conda.yml) 
![Coverage](https://gist.githubusercontent.com/jlerat/47c5eaf5c4ac8375b92a548c057d5c24/raw/coverage_badge.svg)


Python toolbox for hydrological data processing.

# What is hydrodiy?
- hydrodiy is a set of tools to perform standard data analysis
- the package is structured around typical tasks: io, data checking,
  statistical analysis, gis processing and plotting

# Installation
- Create a suitable python environment. We recommend using [miniconda](https://docs.conda.io/projects/miniconda/en/latest/) combined with the environment specification provided in the [env\_hydrodiy.yml](env_hydrodiy.yml) file in this repository.
- Git clone this repository and run `pip install .`

# Basic use
```python
import numpy as np
import matplotlib.pyplot as plt
from hydrodiy.plot import violinplot

data = np.random.normal(size=(200, 5))
plt.close('all')
fig, ax = plt.subplots(layout='tight')

# Draw a nice violin plot
vl = violinplot.Violin(data)
vl.draw(ax=ax)

plt.show()
```
A set of examples is provided in the folder [examples](examples).

# License
The source code and documentation of the hydrodiy package is licensed under the
[BSD license](LICENSE).

