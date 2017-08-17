import sys

# Detect python version
PYVERSION = 2
if sys.version_info > (3, 0):
    PYVERSION = 3

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
