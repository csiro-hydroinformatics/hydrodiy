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

    # Add description in command line options
    txt = re.sub("\\[DESCRIPTION\\]", comment, txt)

    # Add paths
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


class StartedCompletedLogger(logging.Logger):
    """ Add context to logging messages via the context attribute """

    def __init__(self, *args, **kwargs):
        self.context = ""
        super(StartedCompletedLogger, self).__init__(*args, **kwargs)

    def started(self):
        self.info("@@@ Process started @@@")

    def completed(self):
        self.info("@@@ Process completed @@@")


class HydrodiyContextualLogger(StartedCompletedLogger):
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
        fmt="%(asctime)s | %(levelname)s | %(message)s", \
        overwrite=True,
        excepthook=True,
        no_duplicate_handler=True,\
        contextual=False, \
        start_message=True, \
        date_fmt="%y-%m-%d %H:%M"):
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
    date_fmt : str
        Date format.

    Returns
    -----------
    logger : logging.Logger
        Logger instance
    """
    logger = logging.getLogger(name)

    # remove all handlers
    logger.handlers = []

    # Set logging level
    if not level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        raise ValueError("{0} not a valid level".format(level))

    logger.setLevel(getattr(logging, level))

    # Set logging format
    ft = logging.Formatter(fmt, date_fmt)

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
    else:
        logger.__class__ = StartedCompletedLogger

    if start_message:
        logger.started()

    return logger


