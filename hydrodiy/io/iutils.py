import sys, os, re
from pathlib import Path

import shlex
import subprocess
import warnings

from datetime import datetime
import logging
import stat

from hydrodiy import PYVERSION

from io import StringIO,BytesIO

import requests

import numpy as np
import pandas as pd

def dict2str(data, prefix=None):
    """ Convert a dict to a string with the format v1[value1]_v2[value2]

    Parameters
    -----------
    data : dict
        Non nested dictionary containing data
    prefix : str
        Prefix to be added at the beginning of the string

    Returns
    -----------
    vars : str
        String containing variables with prefix

    Example
    -----------
    >>> iutils.dict2string({"name":"bob", "phone":2010})

    """
    out = []

    # Add items
    for key in sorted(data):
        out.append("{0}[{1}]".format(key, data[key]))

    out = "_".join(out)

    # Add prefix if needed
    if not prefix is None:
        if prefix!="":
            out = prefix + "_" + out

    return out


def str2dict(source, num2str=True):
    """ Find match in the form v1[value1]_v2[value2] in the
    source string and returns a dict with the value found

    Parameters
    -----------
    source : str
        String to search in
    num2str : bool
        Convert all value to string

    Example
    -----------
    >>> source = "name[bob]_phone[2010]"
    >>> iutils.str2dict(source)

    """

    # Excludes path and file extension
    source = re.sub("\\.[^\\.]+$", "", os.path.basename(source))
    prefix = source

    # Search for pattern match
    out = {}
    search = re.findall("[^_]+\\[[^\\[]+\\]", source)

    for match in search:
        # Get name
        name = re.sub("\\[.*", "", match)

        # Get value
        value = re.sub(".*\\[|\\]", "", match)

        # Remove item from prefix
        prefix = re.sub("_*"+name+"\\[" + value + "\\]", "", prefix)

        # Attempt conversion if required
        if not num2str:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass

        out[name] = value

    return out, prefix


def script_template(filename, comment,
        type="simple",
        author=None, \
        fout=None, fdata=None, fimg=None):
    """ Write a script template

    Parameters
    -----------
    filename : pathlib.Path
        Filename to write the script to
    comment : str
        Comment on purpose of the script
    type : str
        Type of script:
        * simple: script with minimal functionalities
        * plot: plotting script
    author : str
        Script author
    fout : str
        Output folder
    fdata : str
        Data folder
    fimg : str
        Images folder
    Example
    -----------
    >>> iutils.script_template("a_cool_script.py", "Testing", "plot", "Bob Marley")

    """

    if not type in ["simple", "plot"]:
        raise ValueError("Expected script type in [simple/plot], "+\
                            f"got {type}.")

    # Open script template
    ftemplate = Path(__file__).resolve().parent / \
                                    f"script_template_{type}.py"
    with ftemplate.open("r") as ft:
        txt = ft.read()

    # Add comment header
    if author is None:
        try:
            author = os.getlogin()
        except:
            author = "unknown"

    meta = "## -- Script Meta Data --\n"
    meta += f"## Author  : {author}\n"
    meta += f"## Created : {datetime.now()}\n"
    meta += f"## Comment : {comment}\n"
    meta += "##\n## ------------------------------\n"
    txt = re.sub("\\[COMMENT\\]", meta, txt)

    # -- Add paths --
    filename = Path(filename)

    # By default, the root folder is the script folder
    froot = "source_file.parent"

    # If there is a scripts folder in the path, we use the
    # parent of this path as root.
    parts = filename.parts
    if "scripts" in parts:
        nlevelup = parts[::-1].index("scripts")
        froot += "".join([".parent"]*nlevelup)

    txt = re.sub("\\[FROOT\\]", froot, txt)

    if fout is None:
        fout = "froot / \"outputs\"\nfout.mkdir(exist_ok=True)\n"
    txt = re.sub("\\[FOUT\\]", fout, txt)

    if fdata is None:
        fdata = "froot / \"data\"\nfdata.mkdir(exist_ok=True)\n"
    txt = re.sub("\\[FDATA\\]", fdata, txt)

    if type == "plot":
        if fimg is None:
            fimg = "froot / \"images\"\nfimg.mkdir(exist_ok=True)\n"
        txt = re.sub("\\[FIMG\\]", fimg, txt)
    else:
        txt = re.sub("fimg = \\[FIMG\\]", "", txt)

    # Write
    with filename.open("w") as fs:
        fs.write(txt)

    # Make it executable for the user
    st = os.stat(filename)
    os.chmod(filename, st.st_mode | stat.S_IEXEC)



class HydrodiyContextualLogger(logging.Logger):
    """ Add context to logging messages via the context attribute """

    def __init__(self, *args, **kwargs):
        self.context = ""
        super(HydrodiyContextualLogger, self).__init__(*args, **kwargs)

    def _log(self, level, msg, args, exc_info=None, extra=None):
        if self.context != "":
            msg = "{{ {0} }} {1}".format(self.context, msg)

        super(HydrodiyContextualLogger, self)._log(\
                        level, msg, args, exc_info, extra)


def get_logger(name, level="INFO", \
        console=True, flog=None, \
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s", \
        overwrite=True,
        excepthook=True,
        no_duplicate_handler=True,\
        contextual=False, \
        start_message=True):
    """ Get a logger object that can handle contextual info

    Parameters
    -----------
    name : str
        Logger name
    level : str
        Logging level.
    console : bool
        Log to console
    flog : str
        Path to log file. If none, no log file is used.
    fmt : str
        Log format
    overwrite : bool
        If true, removes the flog files if exists
    excepthook : bool
        Change sys.excepthook to trap all errors in the logger
        (does not work within IPython)
    no_duplicate_handler : bool
        Avoid duplicating console or flog log handlers
    contextual: bool
        Creates a logger with a context attribute
        to add context between curly braces before message
        (see hydrodiy.io.HydrodiyContextualLogger)
    start_message : bool
        Log a "Process started" message.

    Returns
    -----------
    logger : logging.Logger
        Logger instance
    """
    logger = logging.getLogger(name)

    # Set logging level
    if not level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        raise ValueError("{0} not a valid level".format(level))

    logger.setLevel(getattr(logging, level))

    # Set logging format
    ft = logging.Formatter(fmt)

    # Check handlers
    has_console = False
    has_flog = False
    if no_duplicate_handler:
        for hd in logger.handlers:
            if isinstance(hd, logging.StreamHandler):
                has_console = True

            if isinstance(hd, logging.FileHandler):
                if hd.baseFilename == flog:
                    has_flog = True

    # log to console
    if console and not has_console:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(ft)
        logger.addHandler(sh)

    # log to file
    if not flog is None and not has_flog:
        flog = Path(flog)
        if overwrite:
            try:
                flog.unlink()
            except Exception as err:
                warnings.warn(f"log file not deleted: {err}")

        fh = logging.FileHandler(flog)
        fh.setFormatter(ft)
        logger.addHandler(fh)

    if excepthook:
        def catcherr(exc_type, exc_value, exc_traceback):
            logger.error("Unexpected error", exc_info=(exc_type, \
                        exc_value, exc_traceback))
            sys.__excepthook__(exc_type, exc_value, exc_traceback)

        sys.excepthook = catcherr

    # Close all handlers
    [h.close() for h in logger.handlers]

    # Create the contextual logger
    if contextual:
        # A bit dangerous, but will do for now
        logger.__class__ = HydrodiyContextualLogger
        logger.context = ""

    if start_message:
        logger.info("Process started")

    return logger


def read_logfile(flog, \
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s"):
    """ Reads a log file and process data

    Parameters
    -----------
    flog : str
        Log file path
    fmt : str:
        Logging format

    Returns
    -----------
    logs : pandas.core.DataFrame
        Logging items
    """
    # Check inputs
    if not os.path.exists(flog):
        raise ValueError("File {0} does not exist".format(flog))

    # Build regex
    regex = re.sub("\\)s", ">.*)", re.sub("\\%\\(", "(?P<", fmt))
    regex = re.sub("\\|", "\\|", regex)

    # Open log file
    with open(flog, "r") as fobj:
        loglines = fobj.readlines()

    # Extract log info line by line
    logs = []
    for line in loglines:
        se = re.search(regex, line.strip())
        if se:
            logs.append(se.groupdict())

    logs = pd.DataFrame(logs)

    # Process contextual info
    if "message" in logs:
        context = logs.message.str.findall("(?<=\\{)[^\\}]+(?=\\})")
        context = context.apply(lambda x: "".join(x).strip())
        logs["context"] = context

    return logs



