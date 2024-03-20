""" Module to plot data on an Australia map """

import re, os, json, tarfile
from pathlib import Path
import pkg_resources

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.patheffects as patheff

HAS_PYSHP = False
try:
    import shapefile
    HAS_PYSHP = True
except (ImportError, FileNotFoundError) as err:
    pass

# Decompress australia shoreline shapefile
FDATA = Path(pkg_resources.resource_filename(__name__, "data"))
SHAPEFILES = {
    "ozcoast10m": FDATA / "ne_10m_admin_0_countries_australia.shp", \
    "ozcoast50m": FDATA / "ne_50m_admin_0_countries_australia.shp", \
    "ozstates50m": FDATA / "ne_50m_admin_1_states_australia.shp", \
    "ozdrainage": FDATA / "drainage_divisions_lines_simplified.shp", \
    "ozbasins": FDATA / "rbasin_lines_simplified.shp"
}

# Lat long coordinate boxes for regions in Australia
freg = FDATA / "regions.json"
with open(freg, "r") as fo:
    REGIONS = json.load(fo)

# Capital cities
CAPITAL_CITIES = {
    "Brisbane": [153.026, -27.471], \
    "Melbourne": [144.960, -37.821],\
    "Sydney": [151.206, -33.864], \
    "Canberra": [149.134, -35.299], \
    "Hobart": [147.3265, -42.8818], \
    "Adelaide": [138.60, -34.92833], \
    "Perth": [115.86134, -31.95182]
}


def ozlayer(ax, name, filter_field=None, filter_regex=None, proj=None, \
                fixed_lim=True, *args, **kwargs):
    """ plot Australian geographic layer in axes using data
    from Natural Earth.
    (see https://www.naturalearthdata.com/)

    Parameters
    -----------
    ax : matplotlib.axes
        Axe to draw data on
    name : str
        Name of the layer. Data available are:
        - ozcoast10m, ozcoast50m: Natural Earth coast line at
            10m and 50m resolution respectively.
        - ozstates50m : Natural Earth state boundaries at 50m
            resolution.
        - ozdrainage: Drainage boundaries from Geofabric.
        - ozbasins: Basin boundaries from GA.
    filter_field : str
        Shapefile field to filter on.
    fiter_regex : str
        Regular expression to filter from filtered field.
    proj : pyproj.projProj
        Map projection. Example with transform to GDA94:
        proj = pyproj.Proj("+init=EPSG:3112")
    args, kwargs
        Arguments passed to matplotlib.axes.plot function.
    """
    if not HAS_PYSHP:
        raise ValueError("pyshp package could not be imported")

    # Select shapefile to draw
    if not name in SHAPEFILES:
        names = "|".join(list(SHAPEFILES.keys()))
        raise ValueError(f"Expected name in {names}, got {name}.")

    fshp = str(SHAPEFILES[name])

    # Select axis
    if ax is None:
        ax = plt.gca()

    # Plotting function
    def plotit(x, y, recs):
        # Project
        if not proj is None:
            x, y = np.array([proj(xx, yy) for xx, yy in zip(x, y)]).T

        # Plot
        lines = ax.plot(x, y, "-", *args, **kwargs)

        # Return line object
        return (recs, lines[-1])

    # Get xlim and ylim
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Plot shapefile
    with shapefile.Reader(fshp) as shp_object:
        # Get shapefile fields
        fields = np.array([f[0] for f in shp_object.fields])[1:]

        if not filter_field is None:
            # Filter field
            if not filter_field in fields:
                raise ValueError("Expected filter_field in "+\
                    "/".join(list(fields)) + ", got "+filter_field)

            ifilter = np.where(fields == filter_field)[0][0]
        else:
            ifilter = None

        # Draw polygons
        lines = []
        for shp, rec in zip(shp_object.shapes(), shp_object.records()):
            # Apply filter if needed
            if not ifilter is None:
                if not re.search(filter_regex, rec[ifilter]):
                    continue

            # Records to series
            recs = pd.Series(rec, index=fields)

            # Get polygon coordinates
            x, y = np.array(shp.points).T

            if hasattr(shp, "parts"):
                if len(shp.parts) > 1:
                    starts = np.array(shp.parts)
                    ends = np.append(np.array(shp.parts)[1:], len(x))

                    # Plot each part separately
                    for ipart in range(len(shp.parts)):
                        i1, i2 = starts[ipart], ends[ipart]
                        lines.append(plotit(x[i1:i2], y[i1:i2], recs))
                else:
                    # plot
                    lines.append(plotit(x, y, recs))
            else:
                # plot
                lines.append(plotit(x, y, recs))

    if fixed_lim:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    return lines


def ozcities(ax, cities=None, \
                filter_regex=None, \
                fixed_lim=True, plot_kwargs={}, \
                text_kwargs={}, proj=None):
    """ plot Australian capital cities.

    Parameters
    -----------
    ax : matplotlib.axes
        Axe to draw data on
        Shapefile field to filter on.
    cities : dict
        Dictionnary containing cities and their coordinates:
        { city_name: [x, y] }
    fiter_regex : str
        Regular expression to filter city name.
    plot_kwargs : dict
        Argument passed to plot the point representing a city.
    text_kwargs : dict
        Argument passed to write the city name.
        This can be used to adjust label placement. For example,
        the following can be used to add shadow (path effect) and
        offset the labels:

        text_kwargs = dict(
            path_effects=[pe.withStroke(linewidth=3, foreground="w")], \
            textcoords="offset pixels",\
            fontsize=12, \
            xytext=(20, 8)
        )
    proj : pyproj.CRS
        Projec coordinates.

    """
    if cities is None:
        cities = CAPITAL_CITIES

    # Plot options
    plot_kwargs["marker"] = plot_kwargs.get("marker", "s")
    plot_kwargs["mfc"] = plot_kwargs.get("mfc", "tab:orange")
    plot_kwargs["mec"] = plot_kwargs.get("mec", "black")
    plot_kwargs["ms"] = plot_kwargs.get("ms", 7)
    plot_kwargs["color"] = plot_kwargs.get("color", "none")

    text_kwargs["color"] = text_kwargs.get("color", "black")
    pe = [patheff.withStroke(linewidth=2, foreground="w")]
    text_kwargs["path_effects"] = text_kwargs.get("path_effects", pe)

    # Get lims before plotting
    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    elements = {}
    for icity, (city, xy) in enumerate(cities.items()):
        # Skip if filtered
        if not filter_regex is None:
            if not re.search(filter_regex, city):
                continue

        xyproj = xy
        if not proj is None:
            xyproj = proj(*xyproj)

        lab = "Capital city" if icity == 0 else ""
        lines = ax.plot(*xyproj, label=lab, **plot_kwargs)
        txt = ax.annotate(city, xyproj, **text_kwargs)

        # Store
        elements[city] = {"plot": lines[-1], "text": txt}

    if fixed_lim:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    return elements
