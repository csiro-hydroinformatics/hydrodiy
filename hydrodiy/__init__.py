import sys
import importlib

# Detect python version
PYVERSION = 2
if sys.version_info > (3, 0):
    PYVERSION = 3

def has_c_module(name, raise_error=True):
    name = f"c_hydrodiy_{name}"
    out = importlib.util.find_spec(name)

    if not out is None:
        return True
    else:
        if raise_error:
            raise ImportError(f"C module {name} is "+\
                "not available, please run python setup.py build")
        else:
            return False

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
