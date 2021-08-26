import sys

# Detect python version
PYVERSION = 2
if sys.version_info > (3, 0):
    PYVERSION = 3

# C cython modules
HAS_C_DATA_MODULE = True
try:
    import c_hydrodiy_data
except ImportError:
    HAS_C_DATA_MODULE = False

HAS_C_STAT_MODULE = True
try:
    import c_hydrodiy_stat
except ImportError:
    HAS_C_STAT_MODULE = False


HAS_C_GIS_MODULE = True
try:
    import c_hydrodiy_gis
except ImportError:
    HAS_C_GIS_MODULE = False


from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
