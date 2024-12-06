import os, re, json, sys
from pathlib import Path

from string import ascii_letters as letters

import pytest

import warnings

import pandas as pd
import numpy as np

from hydrodiy.io import html

filename = Path(__file__).resolve()
TESTS_DIR = filename.parent

def test_dataframe2html():
    df = pd.DataFrame(np.random.uniform(size=(10, 3)), \
                        columns=["a", "b", "c"])

    fhtml = TESTS_DIR / "dataframe2html_short.html"
    title = "Bidule"
    comment = "Truc"
    author = "Bob Marley"
    html.dataframe2html(df, fhtml, title, comment, author)

    values = np.random.uniform(size=(1000))
    values[np.random.choice(np.arange(1000), 50)] = np.nan
    values = values.reshape((100, 10))
    df = pd.DataFrame(values,  columns=list(letters[:10]))
    df = df.apply(lambda x: x.apply(lambda y: f"{y:0.2g}" if ~np.isnan(y) else "-"))
    day = pd.date_range("2001-01-01", freq="D", periods=100)
    df.loc[:, "Day"] = day
    fhtml = TESTS_DIR / "dataframe2html_big.html"
    title = "Bidule"
    comment = "Truc"
    author = "Bob Marley"
    html.dataframe2html(df, fhtml, title, comment, author)

