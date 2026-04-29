import sys
import importlib

def has_c_module(name, raise_error=True):
    name = f"c_hydrodiy_{name}"
    out = importlib.util.find_spec(name)

    if out is not None:
        return True
    else:
        if raise_error:
            raise ImportError(f"C module {name} is "
                              + "not available, please "
                              + "run python setup.py build")
        else:
            return False
