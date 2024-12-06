import os, re
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import zipfile

from hydrodiy.io import zipjson

SOURCE_FILE = Path(__file__).resolve()
FTEST = Path(__file__).parent
DATA = {"key1": 1, "key2": "this is a string", \
                        "key3": [0.4, 32.2, 12.45]}

def test_write_error():
    """ Test zipjson writer errors """
    filename = FTEST / "zipjson_test2.zip"
    SOURCE_FILE = "bidule"
    comment = "test"
    msg = "Source file"
    with pytest.raises(ValueError, match=msg):
        zipjson.write_zipjson(DATA, filename, comment, \
                            SOURCE_FILE, indent=4)

    msg = "Expected file extension to be .zip"
    f = filename.parent / f"{filename.stem}.bb"
    with pytest.raises(ValueError, match=msg):
        zipjson.write_zipjson(DATA, f, comment, \
                            SOURCE_FILE, indent=4)

def test_read_error():
    """ Test zipjson reader error """
    filename = FTEST / "zipjson_error.zip"
    msg = "No data"
    with pytest.raises(ValueError, match=msg):
        data, meta = zipjson.read_zipjson(filename)


def test_read():
    """ Test zipjson reader """
    filename = FTEST / "zipjson_test.zip"
    data, meta = zipjson.read_zipjson(filename)
    assert data==DATA


def test_write():
    """ Test zipjson writer """
    filename = FTEST / "zipjson_test2.zip"
    comment = "test"
    zipjson.write_zipjson(DATA, filename, comment, \
                            SOURCE_FILE, indent=4)
    data, meta = zipjson.read_zipjson(filename)
    assert data==DATA
    filename.unlink()

