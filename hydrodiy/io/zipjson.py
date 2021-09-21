""" Module to read and write csv data with comments """

import sys, os, re, json
from pathlib import Path
from datetime import datetime
import zipfile
import warnings

from hydrodiy import PYVERSION

from io import StringIO

class ZipjsonWriteError(ValueError):
    """ Error raised by write_zipjson """
    pass

class ZipjsonReadError(ValueError):
    """ Error raised by write_zipjson """
    pass


def write_zipjson(data, filename, \
            comment, source_file, \
            author=None, \
            compression=zipfile.ZIP_DEFLATED, \
            encoding="utf-8", \
            **kwargs):
    """ write a python object to a compressed json file

    Parameters
    -----------
    data : object
        Serializable python object
    filename : pathlib.Path
        Path to file
    comment : str (or dict)
        Comments to be added to header
    source_file : str
        Path to script used to generate the data
    author : str
        Data author (default is given by os.getlogin)
    compression : str
        Compression level
    encoding : str
        Encoding scheme.
    kwargs : dict
        Arguments passed to json.dump
    """
    filename = Path(filename)
    if not filename.suffix == ".zip":
        raise ValueError("Expected file extension to be .zip, "+\
                        f"got {filename.suffix}")

    # Create dict object
    towrite = {"data": data}

    # Add file meta data
    if author is None:
        try:
            author = os.getlogin()
        except Exception:
            author = "unknown"

    towrite["author"] = author
    towrite["comment"] = comment
    towrite["filename"] = str(filename)

    source_file = Path(source_file)
    if not source_file.exists():
        raise ZipjsonWriteError(\
                f"Source file {source_file} does not exists")

    towrite["source_file"] = str(source_file)

    time = datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")
    towrite["time_generated"] = time

    # Zip the file
    with zipfile.ZipFile(str(filename), "w", \
                compression=compression) as archive:
        f = filename.stem + ".json"
        with archive.open(f, "w") as fo:
                    fo.write(json.dumps(towrite, **kwargs)\
                                    .encode(encoding))


def read_zipjson(filename, encoding="utf-8", **kwargs):
    """ Read a json zipped file

    Parameters
    -----------
    filename : str
        Path to file
    encoding : str
        Charactert encoding
    kwargs : dict
        Arguments passed to json.load
    """
    data = {}
    nfiles = 0
    filename = Path(filename)
    with zipfile.ZipFile(str(filename), "r") as archive:
        namelist = archive.namelist()
        if len(namelist) > 1:
            raise ValueError(("Expected one file in zip file {0}, "+\
                        "got {1}").format(filename, len(namelist)))

        fname = namelist[0]
        uni = str(archive.read(fname), encoding=encoding)
        with StringIO(uni) as fobj:
            meta = json.load(fobj, **kwargs)

    if not "data" in meta:
        raise  ValueError("No data in json file")
    else:
        data = meta["data"]

    meta.pop("data")
    if len(meta) == 0:
        warnings.warn("No meta-data in json file")

    return data, meta

