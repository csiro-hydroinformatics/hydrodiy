""" Module to read and write csv data with comments """

import sys
import os
import re
from pathlib import Path, PurePosixPath

from datetime import datetime
from getpass import getuser

import gzip
import zipfile
import pandas as pd
import numpy as np

from io import StringIO

# Check distutils available for python folders
HAS_DISTUTILS = False
try:
    from distutils.sysconfig import get_python_inc
    from distutils.sysconfig import get_python_lib
    HAS_DISTUTILS = True
except ImportError:
    pass

# Max length of comment keys
KEY_LENGTH_MAX = 30


def _header2comment(header):
    """ Format header data into a comment string """
    comment = {}
    i = 1
    for elem in header:
        if not bool(re.search("-{10}", elem)):
            key = re.sub(":.*$", "", elem)
            val = elem[len(key)+1:].strip()
            key = re.sub(" +", "_", key.strip().lower())

            if not bool(re.search(":", elem[:KEY_LENGTH_MAX])):
                key = "comment_{0:02d}".format(i)
                val = elem
                i += 1

            if val != "":
                comment[key] = val

    return comment


def _csvhead(nrow, ncol, comment, source_file,
             write_sys_info, author=None):
    """ Produces a nice header for csv files """

    # Generate the comments dict
    if isinstance(comment, str):
        comments = {"comment": comment}

    elif isinstance(comment, list):
        comments = {}
        for i, com in enumerate(comment):
            comments[f"comment{i:02d}"] = com

    elif isinstance(comment, dict):
        comments = {}
        for k in comment:
            comments[re.sub(":", "", k).lower()] = comment[k]

    else:
        comment = list(comment)
        comments = {}
        for i, com in enumerate(comment):
            comments[f"comment{i:02d}"] = com

    # Generate file header
    head = []
    head.append("# --------------------------------------------------")
    head.append(f"# nrow : {nrow}")
    head.append(f"# ncol : {ncol}")

    for key in sorted(comments):
        head.append(f"# {key} : {comments[key]}")

    now = datetime.now()
    head.append("# time_generated : "
                + f"{datetime.strftime(now, '%Y-%m-%d %H:%M:%S')}")

    # seek author
    if author is None:
        try:
            author = getuser() if write_sys_info else "unknown"
        except Exception:
            author = "unknown"

    head.append("# author : " + author)

    if write_sys_info:
        head.append(f"# source_file : {source_file}")

        # Python config
        head.append(f"# work_dir : {os.getcwd()}")
        head.append(f"# python_environment {os.name}")
        version = sys.version.replace("\n", " ")
        head.append(f"# python_version : {version}")
        head.append(f"# pandas_version : {pd.__version__}")
        head.append(f"# numpy_version : {np.__version__}")

        if HAS_DISTUTILS:
            head.append("# python_inc : " + get_python_inc())
            head.append("# python_lib : " + get_python_lib())
    else:
        head.append(f"# source_file : {source_file.name}")

    head.append("# --------------------------------------------------")

    return head


def _check_name(filename):
    """ Check the name and add the relevant extension """
    filename = Path(filename)

    if filename.exists():
        return filename

    for extension in ["gz", "zip", "csv", "csv.gz"]:
        filename_full = filename.parent / f"{filename.stem}.{extension}"
        if filename_full.exists():
            break

    if not filename_full.exists():
        errmess = f"Cannot find valid file corresponding to {filename}."
        raise ValueError(errmess)

    return filename_full


def write2zip(archive, arcname, txt):
    """ Write txt to fcsv file in zip archive """

    # Check file is not in the archive
    zinfo_test = archive.NameToInfo.get(arcname)
    if zinfo_test is not None:
        raise ValueError(f"File {arcname} already exists in archive")

    # Create fileinfo object
    zinfo = zipfile.ZipInfo(arcname)

    now = datetime.now()
    zinfo.date_time = (now.year, now.month, now.day,
                       now.hour, now.minute, now.second)
    # File permission
    zinfo.external_attr = 33488896  # 0777 << 16

    # Write to archive with header
    archive.writestr(zinfo, txt,
                     compress_type=zipfile.ZIP_DEFLATED)


def write_csv(data, filename, comment,
              source_file,
              author=None,
              write_index=False,
              compress=True,
              float_format="%0.5f",
              archive=None,
              write_sys_info=True,
              **kwargs):
    """ write a pandas dataframe to csv with comments in header

    Parameters
    -----------
    data : pandas.DataFrame
        Dataframe to be written to file
    filename : str or pathlib.Path
        Path to file
    comment : str (or dict)
        Comments to be added to header
    source_file : str or pathlib.Path
        Path to script used to generate the data
    author : str
        Data author (default is given by os.getlogin)
    write_index : bool
        Write dataframe index or not (default not)
    compress : bool
        Compress data to gzip format
    float_format : str
        Floating point number format
    archive : tarfile.TarFile
        Archive to which data is to be added.
    write_sys_info : bool
        Add system info to header.
    kwargs : dict
        Arguments passed to pd.DataFrame.to_csv

    Example
    -----------
    >>> df = pd.DataFrame(np.random.uniform(0, 1, (100, 4))
    >>> # Create an empty source file
    >>> fo = open("script.py", "w"); fo.close()
    >>> # Store data
    >>> csv.write_csv(df, "data.csv", "This is a test", "script.py")
    """
    # Check inputs
    data = pd.DataFrame(data)
    filename = Path(filename)
    source_file = Path(source_file)

    # Generate head
    head = _csvhead(data.shape[0], data.shape[1], comment,
                    source_file=source_file,
                    write_sys_info=write_sys_info,
                    author=author)

    # Check source_file exists
    if not source_file.exists():
        raise ValueError(f"File {source_file} does not exists.")

    if archive is not None:
        compress = False

    # defines file name
    filename_full = filename
    if compress and not filename.suffix == ".zip":
        filename_full = filename.parent / f"{filename.stem}.zip"

    # Open pipe depending on file type
    if archive or compress:
        # No pipe, store dataframe in a string
        # and then write it to archive
        fobj = None
    else:
        fobj = open(filename_full, "w")

        # Write header
        for line in head:
            fobj.write(line+"\n")

    # Write data itself
    txt = data.to_csv(fobj, index=write_index,
                      float_format=float_format, **kwargs)

    # Store compressed data
    if archive or compress:
        txt = "\n".join(head) + "\n" + txt

        # If compress argument, create a zip file
        arcname = str(PurePosixPath(filename))
        if compress:
            arcname = filename.name
            archive = zipfile.ZipFile(filename_full, mode="w",
                                      compression=zipfile.ZIP_DEFLATED)

        write2zip(archive, arcname, txt)
        if compress:
            archive.close()
    else:
        fobj.close()


def read_csv(filename, has_colnames=True, archive=None,
             encoding="utf-8", **kwargs):
    """ Read a pandas dataframe from a csv with comments in header

    Parameters
    -----------
    filename : str
        Path to file
    has_colnames : bool
        Are column names stored in the first line of the data ?
    archive : tarfile.TarFile
        Archive to which data is to be added.
    encoding : str
        Charactert encoding
    kwargs : dict
        Arguments passed to pd.read_csv

    Example
    -----------
    >>> df = pd.DataFrame(np.random.uniform(0, 1, (100, 4))
    >>> # Create an empty script
    >>> fo = open("script.py", "w"); fo.close()
    >>> # Store data
    >>> csv.write_csv(df, "data.csv", "This is a test", "script.py")
    >>> df2 = csv.read_csv("data.csv")

    """
    filename = Path(filename) if archive is None else\
        PurePosixPath(filename)

    if archive is None:
        try:
            # Add gz or zip if file does not exists
            filename_full = _check_name(filename)

            # Open proper file type
            if filename_full.suffix == ".gz":
                with gzip.open(filename_full, "rb") as gzipfile:
                    uni = str(gzipfile.read(), encoding=encoding)
                    fobj = StringIO(uni)

            elif filename_full.suffix == ".zip":
                # Extract the data from archive
                # assumes that file is stored at base level
                with zipfile.ZipFile(filename_full, "r") as zarchive:
                    fcsv = f"{filename.stem}.csv"
                    uni = str(zarchive.read(fcsv), encoding=encoding)
                    fobj = StringIO(uni)

            else:
                fobj = open(filename_full, "r", encoding=encoding)

        except TypeError:
            import warnings
            warnmess = f"Failed to open {filename} as a text file."\
                       + "Will now try to open it as a buffer."
            warnings.warn(warnmess, stacklevel=2)
            fobj = filename

    else:
        # Use the archive mode
        uni = str(archive.read(str(filename)), encoding=encoding)
        fobj = StringIO(uni)

    # Check fobj is readable
    if not hasattr(fobj, "readline"):
        raise ValueError("File object is not readable.")

    # Reads content
    header = []
    comment = {}

    if has_colnames:
        line = fobj.readline()

        while line.startswith("#"):
            header.append(re.sub("^# *|\n$", "", line))
            line = fobj.readline()

        # Extract comment info from header
        comment = _header2comment(header)

        # Generate column headers
        if "names" not in kwargs:
            # deals with multi-index columns
            # reformat columns like (idx1-idx2-...)
            search = re.findall('"\\([^(]*\\)"', line)
            linecols = line

            if search:
                for cn in search:
                    cn2 = re.sub('\\(|\\)|\'|"|\"', "", cn)
                    cn2 = re.sub(", *", "-", cn2)
                    linecols = re.sub(re.escape(cn), cn2, linecols)

            cns = linecols.strip().split(",")
        else:
            cns = kwargs["names"]
            kwargs.pop("names")

        # Reads data with proper column names
        data = pd.read_csv(fobj, names=cns, encoding=encoding, **kwargs)
        data.columns = [re.sub("\\.", "_", cn) for cn in data.columns]

    else:
        # Reads data frame without header
        data = pd.read_csv(fobj, header=None, encoding=encoding, **kwargs)

    fobj.close()

    return data, comment
