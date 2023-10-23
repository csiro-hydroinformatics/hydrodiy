from pathlib import Path
import pytest
import json

import numpy as np
import pandas as pd

from hydrodiy.gis import gutils
from hydrodiy import has_c_module


FTEST = Path(__file__).resolve().parent

TRIANGLE = np.array([[-1.0, -1.0], [0.0, 1.0], [1.0, 0.0]])

SKIPMESS = "c_hydrodiy_gis module is not available. Please compile."

@pytest.mark.skipif(not has_c_module("gis", False), reason=SKIPMESS)
def test_point_inside_triangle():
    points = np.array([[0.2, 0.2], [1.0, 1.0], [-0.2, -0.2]])
    inside = gutils.points_inside_polygon(points, TRIANGLE)
    expected = np.array([True, False, True])
    assert np.array_equal(inside, expected)

    points = np.array([[-0.2, -0.2], [0.2, 0.2], [1.0, 1.0],
                [1e-30, 1e-30], [10., 10.]])
    inside = gutils.points_inside_polygon(points, TRIANGLE)
    expected = np.array([True, True, False, True, False])
    assert np.array_equal(inside, expected)


@pytest.mark.skipif(not has_c_module("gis", False), reason=SKIPMESS)
def test_point_inside_polygon():
    # Additional data to test points in polygon algorithm
    fp = FTEST / "polygon.csv"
    xy = np.loadtxt(fp, delimiter=",")

    # Define grid
    xlim = xy[:, 0].min(), xy[:, 0].max()
    ylim = xy[:, 1].min(), xy[:, 1].max()
    x = np.linspace(*xlim, 30)
    y = np.linspace(*ylim, 30)
    xx, yy = np.meshgrid(x, y)

    # Compute inside/outside
    points = np.column_stack([xx.flat, yy.flat])
    inside = gutils.points_inside_polygon(points, xy)

    fp = FTEST / "polygon_inside.csv"
    expected = np.loadtxt(fp, delimiter=",").astype(bool)
    assert np.array_equal(inside, expected)


@pytest.mark.skipif(not has_c_module("gis", False), reason=SKIPMESS)
def test_point_inside_polygon_memory_efficient():
    points = np.array([[0.2, 0.2], [1.0, 1.0], [-0.2, -0.2]])
    inside = np.zeros(3, dtype=np.int32)
    gutils.points_inside_polygon(points, TRIANGLE, \
                            inside=inside)
    expected = np.array([True, False, True])
    assert np.array_equal(inside, expected)


@pytest.mark.skipif(not has_c_module("gis", False), reason=SKIPMESS)
def test_point_inside_polygon_memory_efficient_error():
    points = np.array([[0.2, 0.2], [1.0, 1.0], [-0.2, -0.2]])
    inside = np.zeros(2, dtype=np.int32)

    msg = "Expected inside of length"
    with pytest.raises(ValueError, match=msg):
        gutils.points_inside_polygon(points, TRIANGLE, \
                            inside=inside)

    inside = np.zeros(3)
    msg = "Expected inside of dtype"
    with pytest.raises(ValueError, match=msg):
        gutils.points_inside_polygon(points, TRIANGLE, \
                            inside=inside)

    inside = np.array([""]*3)
    msg = "Expected inside of dtype"
    with pytest.raises(ValueError, match=msg):
        gutils.points_inside_polygon(points, TRIANGLE, \
                            inside=inside)


