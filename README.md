# hydrodiy
Python toolbox for hydrological data processing.

# What is hydrodiy?
- hydrodiy is a set of tools to perform standard data analysis
- the package is structured around typical tasks: io, data checking,
  statistical analysis, gis processing and plotting

#Installation
- Create a suitable python environment. We recommend using [miniconda](https://docs.conda.io/projects/miniconda/en/latest/) combined with the environment specification provided in the [env_mini.yml] (env_mini.yml) file in this repository.
- Download the [source code] (https://github.com/csiro-hydroinformatics/hydrodiy) and
- run `(pip install .)`

#Basic use
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
[BSD license](LICENSE.txt).

