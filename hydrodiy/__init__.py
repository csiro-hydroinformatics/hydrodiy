import sys
import imp

# Detect python version
PYVERSION = 2
if sys.version_info > (3, 0):
    PYVERSION = 3

def has_c_module(name, raise_error=True):
    try:
        name = f"c_hydrodiy_{name}"
        fp, pathname, description = imp.find_module(name)
        return True
    except ImportError:
        if raise_error:
            raise ImportError(f"C module c_hydrodiy_{name} is "+\
                "not available, please run python setup.py build")
        else:
            return False

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
