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


class StartedCompletedLogger():
    """ Add context to logging messages via the context attribute """

    def __init__(self, logger, \
                    separator_charac="-", \
                    separator_length=50, \
                    dictseparator_charac="+", \
                    dictseparator_length=30, \
                    tab_length=4):
        errmess = "Expected a logger object"
        assert isinstance(logger, logging.Logger), errmess
        self._logger = logger

        self.separator_charac = separator_charac
        self.separator_length = separator_length

        self.dictseparator_charac = dictseparator_charac
        self.dictseparator_length = dictseparator_length

        self.tab_length = tab_length

    def get_separator(self, nsep, sep):
        return sep*nsep

    def add_tab(self, msg, ntab):
        if ntab==0:
            return msg

        tab_space = " "*self.tab_length*ntab
        return tab_space+msg

    def error(self, msg, ntab=0, *args, **kwargs):
        msg = self.add_tab(msg, ntab)
        return self._logger.error(msg, *args, **kwargs)

    def info(self, msg, ntab=0, *args, **kwargs):
        msg = self.add_tab(msg, ntab)
        return self._logger.info(msg, *args, **kwargs)

    def warning(self, msg, ntab=0, *args, **kwargs):
        msg = self.add_tab(msg, ntab)
        return self._logger.warning(msg, *args, **kwargs)

    def critical(self, msg, ntab=0, *args, **kwargs):
        msg = self.add_tab(msg, ntab)
        return self._logger.critical(msg, *args, **kwargs)

    @property
    def handlers(self):
        return self._logger.handlers

    def started(self):
        self.info("@@@ Process started @@@")
        self.info(self.get_separator(self.separator_charac, \
                                            self.separator_length))
        self.info("")

    def completed(self):
        self.info("")
        self.info(self.get_separator(self.separator_charac, \
                                            self.separator_length))
        self.info("@@@ Process completed @@@")

    def log_dict(self, tolog, name="", level="info"):
        """ Add log entry for dictionnary (e.g. created from argparse using  vars)"""
        assert level in ["info", "warning", "critical", "error"]
        logfun = getattr(self, level)
        sep = self.get_separator(self.dictseparator_charac, \
                                    self.dictseparator_length)
        logfun(sep)
        if name!="":
            logfun(f"{name}:")
        for k, v in tolog.items():
            msg = self.add_tab(f"{k} = {v}", 1)
            logfun(msg)

        logfun(sep)
        logfun("")



class ContextualLogger(StartedCompletedLogger):
    """ Add context to logging messages via the context attribute """

    def __init__(self, logger, \
                        context_hasheader=False, \
                        context_charac="#", \
                        context_length=3, \
                        *args, **kwargs):
        self._context = ""
        self.context_hasheader = context_hasheader
        self.context_charac = context_charac
        self.context_length = context_length

        super(ContextualLogger, self).__init__(logger, *args, **kwargs)

    @property
    def context(self):
        return self._context

    @context.setter
    def context(self, value):
        self._context = ""
        self.info("")
        self._context = str(value)
        if self._context != "" and self.context_hasheader:
            sep = self.get_separator(self.context_charac, \
                                        self.context_length)
            mess = sep+" "+self._context+" "+sep
            self.info(mess)

    def completed(self):
        self.context = ""
        super(ContextualLogger, self).completed()

    def get_message(self, msg, ntab):
        if self.context != "":
            tab = " "*self.tab_length*ntab if ntab>0 else ""
            return "{{ {0} }} {1}{2}".format(self.context, tab, msg)
        return msg

    def error(self, msg, ntab=0, *args, **kwargs):
        msg = self.get_message(msg, ntab)
        return self._logger.error(msg, *args, **kwargs)

    def info(self, msg, ntab=0, *args, **kwargs):
        msg = self.get_message(msg, ntab)
        return self._logger.info(msg, *args, **kwargs)

    def warning(self, msg, ntab=0, *args, **kwargs):
        msg = self.get_message(msg, ntab)
        return self._logger.warning(msg, *args, **kwargs)

    def critical(self, msg, ntab=0, *args, **kwargs):
        msg = self.get_message(msg, ntab)
        return self._logger.critical(msg, *args, **kwargs)


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

    # Create the extended logger
    elogger = ContextualLogger(logger) if contextual else \
                    StartedCompletedLogger(logger)

    if start_message:
        elogger.started()

    return elogger


