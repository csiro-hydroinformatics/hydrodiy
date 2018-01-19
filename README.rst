hydrodiy
========

Python toolbox for hydrological data processing.

What is hydrodiy?
~~~~~~~~~~~~~~~~~

- hydrodiy is a set of tools to perform standard data analysis
- the package is structured around typical tasks: io, data checking,
  statistical analysis, gis processing and plotting

Installation
~~~~~~~~~~~~

``pip install hydrodiy`` or download the `source
code <https://bitbucket.org/jlerat/hydrodiy>`__ and
``python setup.py install``

Basic use
~~~~~~~~~


   .. code:: 

       import numpy as np
       import matplotlib.pyplot as plt
       from hydrodiy.plot import boxplot

       data = np.random.normal(size=(200, 5))
       plt.close('all')
       fig, ax = plt.subplots()
       
       # Draw a nice box plot
       bp = boxplot.Boxplot(data)
       bp.draw()

       # Show sample count 
       bp.show_count()

A set of examples is provided in the folder 'examples'.

License
~~~~~~~~~

The source code and documentation of the hydrodiy package is licensed under the
`GPLv3 license <https://www.gnu.org/licenses/gpl-3.0.en.html>`__.
