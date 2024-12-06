import re
import pytest
from pathlib import Path
import numpy as np
import pandas as pd
import tarfile
import zipfile

from hydrodiy.io import csv

FTEST = Path(__file__).resolve().parent


def test_read_csv1():
    fcsv = FTEST / "states_centroids.csv.gz"
    data, comment = csv.read_csv(fcsv)
    st = pd.Series(["ACT", "NSW", "NT", "QLD", "SA",
                        "TAS", "VIC", "WA"])
    assert (all(data["state"]==st))


def test_read_csv_names():
    fcsv = FTEST / "states_centroids.csv.gz"
    cc = ["C{0}".format(k) for k in range(1, 8)]
    data, comment = csv.read_csv(fcsv, names=cc)
    assert (list(data.columns)==cc)


def test_read_csv_names_noheader():
    fcsv = FTEST / "states_centroids_noheader.csv"
    cc = ["C{0}".format(k) for k in range(1, 8)]
    data, comment = csv.read_csv(fcsv, has_colnames=False, names=cc)
    assert (list(data.columns)==cc)


def test_read_csv_noheader():
    fcsv = FTEST / "states_centroids_noheader.csv"
    data, comment = csv.read_csv(fcsv, has_colnames=False)
    st = pd.Series(["ACT", "NSW", "NT", "QLD", "SA",
                        "TAS", "VIC", "WA"])
    assert (all(data[0]==st))


def test_read_csv3():
    fcsv = FTEST / "multiindex.csv"
    data, comment = csv.read_csv(fcsv)

    cols =["metric", "runoff_rank",
            "logsinh-likelihood", "logsinh-shapirotest",
            "yeojohnson-likelihood", "yeojohnson-shapirotest"]

    assert (all(data.columns==cols))


def test_read_csv4():
    fcsv = FTEST / "climate.csv"
    data, comment = csv.read_csv(fcsv,
            parse_dates=[""], index_col=0)

    assert (len(comment) == 8)
    assert (comment["written_on"] == "2014-08-12 12:41")

    d = data.index[0]
    try:
        assert (isinstance(d, pd.tslib.Timestamp))
    except AttributeError:
        # To handle new versions of pandas
        assert (isinstance(d, pd.Timestamp))


def test_read_csv5():
    fcsv = FTEST / "207004_monthly_total_01.csv"
    data, comment = csv.read_csv(fcsv,
            parse_dates=True, index_col=0)


def test_read_csv_latin():
    """ Test read_csv with latin_1 encoding """
    fcsv = FTEST / "latin_1.zip"
    with pytest.raises(UnicodeDecodeError):
        data, comment = csv.read_csv(fcsv)

    data, comment = csv.read_csv(fcsv,
            encoding="latin_1")
    assert (np.allclose(data.iloc[:, 1:4].values, -99))


def test_write_csv1():
    nval = 100
    nc = 5
    idx = pd.date_range("1990-01-01", periods=nval, freq="D")
    df1 = pd.DataFrame(np.random.normal(size=(nval, nc)), index=idx)

    fcsv1 = FTEST / "testwrite1.csv"
    csv.write_csv(df1, fcsv1, "Random data",
            Path(__file__),
            write_index=True)

    fcsv2 = FTEST / "testwrite2.csv"
    csv.write_csv(df1, fcsv2, "Random data",
            Path(__file__),
            float_format=None,
            write_index=True)

    df1exp, comment = csv.read_csv(fcsv1,
            parse_dates=[""], index_col=0)

    df2exp, comment = csv.read_csv(fcsv2,
            parse_dates=[""], index_col=0)

    assert (int(comment["nrow"]) == nval)
    assert (int(comment["ncol"]) == nc)

    d = df1exp.index[0]
    try:
        assert (isinstance(d, pd.tslib.Timestamp))
    except AttributeError:
        # To handle new versions of Pandas
        assert (isinstance(d, pd.Timestamp))

    assert (np.allclose(np.round(df1.values, 5), df1exp))
    assert (np.allclose(df1, df2exp))

    for f in [fcsv1, fcsv2]:
        fz = f.parent / f"{f.stem}.zip"
        fz.unlink()


def test_write_csv2():
    nval = 100
    nc = 5
    idx = pd.date_range("1990-01-01", periods=nval, freq="D")
    df1 = pd.DataFrame(np.random.normal(size=(nval, nc)), index=idx)

    comment1 = {"co1":"comment", "co2":"comment 2"}
    fcsv = FTEST / "testwrite.csv"
    csv.write_csv(df1, fcsv, comment1,
            author="toto",
            source_file=Path(__file__),
            write_index=True)

    df2, comment2 = csv.read_csv(fcsv,
            parse_dates=[""], index_col=0)

    assert (comment1["co1"] == comment2["co1"])
    assert (comment1["co2"] == comment2["co2"])
    assert ("toto" == comment2["author"])
    assert (str(Path(__file__)) == comment2["source_file"])

    fz = fcsv.parent / f"{fcsv.stem}.zip"
    fz.unlink()


def test_write_csv3():
    nval = 100
    idx = pd.date_range("1990-01-01", periods=nval, freq="D")
    ds1 = pd.Series(np.random.normal(size=nval), index=idx)

    fcsv1 = FTEST / "testwrite3.csv"
    csv.write_csv(ds1, fcsv1, "Random data",
            Path(__file__),
            write_index=True)

    ds1exp, comment = csv.read_csv(fcsv1,
            parse_dates=[""], index_col=0)
    ds1exp = ds1exp.squeeze()

    assert (np.allclose(ds1.round(5), ds1exp))

    fz = fcsv1.parent / f"{fcsv1.stem}.zip"
    fz.unlink()


def test_read_write_zip():
    # Generate data
    df = {}
    for i in range(4):
        df["test_{0:02d}/test_{0}.csv".format(i)] = \
                pd.DataFrame(np.random.normal(size=(100, 4)))

    # Write data to archive
    farc = FTEST / "test_archive.zip"
    with zipfile.ZipFile(farc, "w") as arc:
        for k in df:
            # Add file to tar with a directory structure
            csv.write_csv(df[k],
                filename=k,
                comment="test "+str(i),
                archive=arc,
                float_format="%0.20f",
                source_file=Path(__file__))

    # Read it back and compare
    with zipfile.ZipFile(farc, "r") as arc:
        for k in df:
            df2, _ = csv.read_csv(k, archive=arc)
            assert (np.allclose(df[k].values, df2.values))

    farc.unlink()

