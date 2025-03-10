import re
import math
import numpy as np
import pandas as pd

from matplotlib import lines, colors as mcolors
from matplotlib.patches import FancyBboxPatch
import matplotlib.pyplot as plt

COLORS = [v for k, v in mcolors.TABLEAU_COLORS.items()]
EPS = 1e-10


class BoxplotError(Exception):
    pass


def compute_percentiles(coverage):
    """ Compute whiskers percentiles from coverage """
    qq1 = float(100-coverage)/2
    qq2 = 100.-qq1
    return qq1, qq2


def boxplot_stats(data, box_coverage, whiskers_coverage):
    """ Compute boxplot stats """

    idx = (~np.isnan(data)) & (~np.isinf(data))
    bqq1, bqq2 = compute_percentiles(box_coverage)
    wqq1, wqq2 = compute_percentiles(whiskers_coverage)
    nok = np.sum(idx)

    if nok > 3:
        qq = [wqq1, bqq1, 50, bqq2, wqq2]
        prc = pd.Series(np.nanpercentile(data[idx], qq),
                        index=["{0:0.1f}%".format(qqq) for qqq in qq])
        prc["count"] = nok
        prc["mean"] = data[idx].mean()
        prc["max"] = data[idx].max()
        prc["min"] = data[idx].min()
    else:
        prc = pd.Series({"count": nok})
        pnames = ["{0:0.1f}%".format(wqq1),
                  "{0:0.1f}%".format(bqq1),
                  "50.0%",
                  "{0:0.1f}%".format(bqq2),
                  "{0:0.1f}%".format(wqq2),
                  "min", "max", "mean"]
        for pn in pnames:
            prc[pn] = np.nan

    return prc


def _to_float(value, mini=0., maxi=np.inf):
    """ Convert value to float with range checking """
    try:
        value = np.float64(value)
    except Exception:
        raise BoxplotError(f"Failed to convert {value} to float.")

    if value < mini-EPS or value > maxi+EPS:
        errmsg = f"Expected value in [{mini}, {maxi}], got {value}."
        raise BoxplotError(errmsg)

    return value


def _is_in(value, possible):
    """ Check if a value is in a set of possible choices """
    if value not in possible:
        errmsg = f"Expected value in {possible}, got {value}."
        raise BoxplotError(errmsg)

    return value


class BoxplotItem(object):
    """ Element of the boxplot graph """

    def __init__(self,
                 linestyle="-",
                 alpha=1.,
                 linewidth=2,
                 linecolor="k",
                 facecolor="none",
                 width=0.2,
                 va="center",
                 ha="left",
                 fontcolor="k",
                 number_format="%0.1f",
                 fontsize=10,
                 fontweight="normal",
                 boxstyle="Square,pad=0",
                 hatch="none",
                 marker="o",
                 markerfacecolor="none",
                 markeredgecolor="k",
                 markersize=5,
                 show_line=True,
                 show_text=True):

        # Attributes controlled by getters/setters
        self._linestyle = None
        self._alpha = None
        self._va = None
        self._ha = None
        self._marker = None
        self._boxstyle = None
        self._linewidth = None
        self._fontsize = None
        self._fontweight = None
        self._markersize = None
        self._width = None
        self._hatch = None

        # Set attributes
        self.linestyle = linestyle
        self.alpha = alpha
        self.va = va
        self.ha = ha
        self.marker = marker
        self.boxstyle = boxstyle
        self.hatch = hatch

        self.linewidth = linewidth
        self.fontsize = fontsize
        self.fontweight = fontweight
        self.markersize = markersize
        self.width = width

        self.linecolor = linecolor
        self.facecolor = facecolor
        self.fontcolor = fontcolor
        self.markerfacecolor = markerfacecolor
        self.markeredgecolor = markeredgecolor

        self.number_format = number_format

        self.show_line = show_line
        self.show_text = show_text

    @property
    def linestyle(self):
        return self._linestyle

    @linestyle.setter
    def linestyle(self, value):
        self._linestyle = _is_in(value, lines.lineStyles.keys())

    @property
    def va(self):
        return self._va

    @va.setter
    def va(self, value):
        va = ["top", "center", "bottom", "baseline"]
        self._va = _is_in(value, va)

    @property
    def ha(self):
        return self._ha

    @ha.setter
    def ha(self, value):
        ha = ["left", "center", "right"]
        self._ha = _is_in(value, ha)

    @property
    def marker(self):
        return self._marker

    @marker.setter
    def marker(self, value):
        markers = ["none", "o", ".", "*", "+"]
        self._marker = _is_in(value, markers)

    @property
    def boxstyle(self):
        return self._boxstyle

    @boxstyle.setter
    def boxstyle(self, value):
        if not re.search("^(Square|Round)", value):
            errmsg = "Boxstyle should start either by Round or"\
                     + f" Square, got {value}."
            raise BoxplotError(errmsg)
        self._boxstyle = value

    @property
    def linewidth(self):
        return self._linewidth

    @linewidth.setter
    def linewidth(self, value):
        self._linewidth = _to_float(value)

    @property
    def fontsize(self):
        return self._fontsize

    @fontsize.setter
    def fontsize(self, value):
        self._fontsize = _to_float(value)

    @property
    def fontweight(self):
        return self._fontweight

    @fontweight.setter
    def fontweight(self, value):
        self._fontweight = _is_in(str(value), ["normal", "bold"])

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        self._width = _to_float(value)

    @property
    def markersize(self):
        return self._markersize

    @markersize.setter
    def markersize(self, value):
        self._markersize = _to_float(value)

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = _to_float(value, maxi=1.)

    @property
    def hatch(self):
        return self._hatch

    @hatch.setter
    def hatch(self, value):
        hatches = ["none", "/",
                   "\\", "\\\\", "\\\\\\", "//", "///", "|", "-",
                   "+", "x", "o", "O", ".", "*"]
        self._hatch = _is_in(str(value), hatches)


class Boxplot(object):
    """ Object allowing to draw boxplots """

    def __init__(self, data,
                 style="default", by=None,
                 show_mean=False,
                 show_median=True,
                 show_text=False,
                 center_text=True,
                 linewidth=2,
                 width_from_count=False,
                 number_format="0.2f",
                 box_coverage=50.,
                 whiskers_coverage=90.):
        """ Draw boxplots with labels and defined colors

        Parameters
        -----------
        data : pandas.Series, pandas.DataFrame, numpy.ndarray
            Data to be plotted
        style : str
            Boxplot style. Possible:
            * default : Standard boxplot
            * narrow : Boxplot for large amount of data
        by : pandas.Series
            Categories used to split data
        show_mean : bool
            Display mean as a dot
        show_median : bool
            Display median as a line
        show_text : bool
            Display summary statistics values
        show_text : bool
            Display summary statistics values
        center_text : bool
            Center the text within the boxplot instead of on the side
        linewidth : int
            Plot lines width in points
        width_from_count : bool
            Use counts to define the boxplot width
        number_format : int
            Number of digits in number format
        box_coverage : float
            Coverage defining the percentile used to compute box extent.
            Example: coverage = 50% -> 25%/75% whiskers
        whiskers_coverage : float
            Coverage defining the percentile used to compute whiskers extent.
            Example: coverage = 90% -> 5%/95% whiskers
        """

        # Check input data
        if by is not None:
            by = pd.Series(by)
            if by.name is None:
                by.name = "by"

            if len(by.unique()) == 1:
                raise BoxplotError("by has 1 category only")

            try:
                data = pd.Series(data).astype(np.float64)
            except Exception as err:
                errmsg = f"Failed to convert data to float series: {err}."
                raise BoxplotError(errmsg)

        else:
            try:
                data = pd.DataFrame(data).astype(np.float64)
            except Exception as err:
                errmsg = "Failed to convert data to float dataframe:"\
                         + f" {err}"
                raise BoxplotError(errmsg)

        # initialise objects
        self._ax = None
        self._data = data
        self._by = by
        self.elements = None
        self.median = None
        self.mean = None

        if box_coverage < 40.:
            errmsg = "Bor coverage cannot be below 40."\
                     + f" Got {box_coverage}."
            raise BoxplotError(errmsg)
        self.box_coverage = box_coverage

        if whiskers_coverage <= box_coverage:
            errmsg = "Whiskers coverage cannot be below box coverage."\
                     + f" Got {box_coverage} for box cov and"\
                     + f"{whiskers_coverage} for whisk cov."
            raise BoxplotError(errmsg)
        self.whiskers_coverage = whiskers_coverage

        self._width_from_count = width_from_count

        # Configure box plot formats depending on style
        if style == "default":
            if show_median:
                self.median = BoxplotItem(linecolor=COLORS[3],
                                          fontcolor=COLORS[3],
                                          fontsize=9,
                                          marker="none",
                                          linewidth=linewidth,
                                          show_text=show_text)

            if show_mean:
                self.mean = BoxplotItem(marker="o",
                                        markerfacecolor=COLORS[4],
                                        markeredgecolor=COLORS[4],
                                        markersize=6,
                                        show_line=False,
                                        fontcolor=COLORS[4],
                                        show_text=False)

            self.whiskers = BoxplotItem(linecolor=COLORS[0],
                                        facecolor=COLORS[0],
                                        linewidth=linewidth,
                                        width=0.)

            self.caps = BoxplotItem(linecolor=COLORS[0],
                                    linewidth=linewidth,
                                    width=0.3, fontcolor=COLORS[0])

            self.box = BoxplotItem(linecolor=COLORS[0],
                                   width=0.7,
                                   fontcolor=COLORS[0],
                                   number_format=f"%0.{number_format}f",
                                   fontsize=8,
                                   linewidth=linewidth,
                                   show_text=False)

        elif style == "narrow":
            obj = BoxplotItem(marker="o",
                              markeredgecolor=COLORS[0],
                              markerfacecolor="w",
                              show_line=False,
                              show_text=False)
            if show_median:
                self.median = obj
            else:
                self.mean = obj

            self.whiskers = BoxplotItem(linecolor="none",
                                        facecolor=COLORS[0],
                                        alpha=0.5,
                                        width=0.3, linewidth=0)

            self.caps = BoxplotItem(linecolor="none", width=0.)

            self.box = BoxplotItem(linecolor="none",
                                   facecolor=COLORS[0],
                                   width=0.6, show_text=False)
        else:
            errmsg = f"Expecting style in [default/narrow], got {style}."
            raise BoxplotError(errmsg)

        # Set text format
        for obj in [self.median, self.mean, self.whiskers,
                    self.box, self.caps]:
            if obj is not None:
                obj.number_format = number_format

        self.center_text = center_text
        if center_text:
            for obj in [self.median, self.mean, self.box, self.whiskers]:
                if obj is not None:
                    obj.ha = "center"
                    obj.va = "bottom"
            self.box.ha = "left"

        # Items not affected by style
        self.count = BoxplotItem(fontsize=7,
                                 fontcolor="grey",
                                 number_format="%d")

        self.minmax = BoxplotItem(markerfacecolor=COLORS[0],
                                  marker="none",
                                  show_line=False)

        # Compute boxplot stats
        self._compute()

    def _compute(self):
        """ Compute boxplot stats """

        data = self._data
        by = self._by
        bhc = self.box_coverage
        whc = self.whiskers_coverage

        if by is None:
            self._stats = data.apply(boxplot_stats, args=(bhc, whc, ))
        else:
            stats = data.groupby(by).apply(boxplot_stats, bhc, whc)

            # Reformat to make a 2d dataframe
            stats = stats.reset_index()
            self._stats = pd.pivot_table(stats,
                                         index="level_1",
                                         columns=by.name,
                                         values=stats.columns[-1])

    @property
    def stats(self):
        """ Returns the boxplot stats """
        return self._stats

    @property
    def ax(self):
        """ Returns the boxplot axe """
        return self._ax

    def draw(self, ax=None, logscale=False, xoffset=0.):
        """ Draw the boxplot

        Parameters
        -----------
        ax : matplotlib.axes
            Axe to draw the boxplot on
        logscale : bool
            Use y axis log scale or not
        xoffset : float
            Add an offset to x axis
        """

        if ax is None:
            self._ax = plt.gca()
        else:
            self._ax = ax

        ax = self._ax
        stats = self._stats
        ncols = stats.shape[1]

        bqq1, bqq2 = compute_percentiles(self.box_coverage)
        bqq1txt = "{0:0.1f}%".format(bqq1)
        bqq2txt = "{0:0.1f}%".format(bqq2)

        wqq1, wqq2 = compute_percentiles(self.whiskers_coverage)
        wqq1txt = "{0:0.1f}%".format(wqq1)
        wqq2txt = "{0:0.1f}%".format(wqq2)

        # Boxplot widths
        if self._width_from_count:
            cnt = stats.loc["count", :].values
            ratio = cnt/cnt.max()
            boxwidths = ratio*self.box.width
            capswidths = ratio*self.caps.width
            whiskerswidths = ratio*self.whiskers.width
        else:
            boxwidths = np.ones(ncols)*self.box.width
            capswidths = np.ones(ncols)*self.caps.width
            whiskerswidths = np.ones(ncols)*self.whiskers.width

        # Loop through stats
        self.elements = {}
        for i, colname in enumerate(stats.columns):
            # initialise boxplot elements
            element = {}

            # Box Widths
            bw = boxwidths[i]

            # Draw median and mean
            x = [i-bw/2+xoffset, i+bw/2+xoffset]
            for statname in ["median", "mean"]:
                stn = "50.0%" if statname == "median" else "mean"
                value = stats.loc[stn, colname]
                y = [value] * 2
                valid_value = np.all(~np.isnan(value))

                item = getattr(self, statname)
                if item is not None:
                    if item.show_line and valid_value:
                        ax.plot(x, y, lw=item.linewidth,
                                color=item.linecolor,
                                alpha=item.alpha)
                        element[statname+"-line"] = ax.get_lines()[-1]

                    if item.marker != "none":
                        ax.plot(i+xoffset, value, marker=item.marker,
                                markeredgecolor=item.markeredgecolor,
                                markerfacecolor=item.markerfacecolor,
                                markersize=item.markersize,
                                alpha=item.alpha)
                        element[statname+"-marker"] = ax.get_lines()[-1]

                    if item.show_text and valid_value:
                        if item.ha == "left":
                            valuetext = f" {value:{item.number_format}}"
                            xshift = bw/2
                        else:
                            valuetext = f"{value:{item.number_format}}"
                            xshift = 0

                        txt = ax.text(i+xshift+xoffset,
                                      value,
                                      valuetext,
                                      fontsize=item.fontsize,
                                      color=item.fontcolor,
                                      va=item.va, ha=item.ha,
                                      alpha=item.alpha)
                        element[statname+"-text"] = txt

            # Skip missing data
            q1 = stats.loc[bqq1txt, colname]
            q2 = stats.loc[bqq2txt, colname]

            valid_q1 = np.all(~np.isnan(q1))
            valid_q2 = np.all(~np.isnan(q2))
            if not valid_q1 or not valid_q2:
                continue

            # Draw box
            item = self.box
            if item.facecolor != "none" or item.show_line:
                # Remove rounding to avoid weird spikes
                # when there are too many columns
                boxstyle = item.boxstyle
                if ncols > 8 or logscale:
                    boxstyle = re.sub("rounding_size=[^,]+",
                                      "rounding_size=0.", boxstyle)

                bbox = FancyBboxPatch([i-bw/2+xoffset, q1], bw, q2-q1,
                                      boxstyle=boxstyle,
                                      facecolor=item.facecolor,
                                      linewidth=item.linewidth,
                                      edgecolor=item.linecolor,
                                      alpha=item.alpha)
                ax.add_patch(bbox)
                element["box"] = ax.patches[-1]

            # Whisker width
            ww = whiskerswidths[i]

            # Draw whiskers
            item = self.whiskers
            if item.show_line:
                wnames = [[wqq1txt, bqq1txt], [wqq2txt, bqq2txt]]
                for icc, cc in enumerate(wnames):
                    # Get y data
                    q1 = stats.loc[cc[0], colname]
                    q2 = stats.loc[cc[1], colname]

                    if ww < EPS:
                        # Draw line
                        x = [i+xoffset]*2
                        y = [q1, q2]
                        ax.plot(x, y, lw=item.linewidth,
                                color=item.linecolor,
                                alpha=item.alpha)

                        element[f"bottom-whiskers{icc+1}"] = ax.get_lines()[-2]
                        element[f"top-whiskers{icc+1}"] = ax.get_lines()[-1]
                    else:
                        # Draw box
                        bbox = FancyBboxPatch([i-ww/2+xoffset, q1],
                                              ww, q2-q1,
                                              boxstyle=item.boxstyle,
                                              facecolor=item.facecolor,
                                              linewidth=item.linewidth,
                                              edgecolor=item.linecolor,
                                              alpha=item.alpha)
                        ax.add_patch(bbox)
                        element[f"whiskers{icc+1}"] = ax.patches[-1]

            # Cap width
            cw = capswidths[i]

            # Draw caps
            item = self.caps
            if item.show_line and cw > 0:
                for iqq, qq in enumerate([wqq1txt, wqq2txt]):
                    q1 = stats.loc[qq, colname]
                    x = [i-cw/5+xoffset, i+cw/5+xoffset]
                    y = [q1]*2
                    ax.plot(x, y, lw=item.linewidth,
                            color=item.linecolor,
                            alpha=item.alpha)

                    element[f"bottom-cap{iqq+1}"] = ax.get_lines()[-2]
                    element[f"top-cap{iqq+1}"] = ax.get_lines()[-1]

            # Box (quartile) values
            item = self.box
            if item.show_text:
                # Define formatter
                formatter = item.number_format
                if item.ha == "left":
                    formatter = " "+formatter
                    xshift = bw/2
                elif item.ha == "center":
                    xshift = 0

                values = [stats.loc[qq, colname] for qq in [bqq1txt, bqq2txt]]

                for ivalue, value in enumerate(values):
                    va, ha = item.va, item.ha
                    if item.ha == "left":
                        valuetext = f" {value:{item.number_format}}"
                    else:
                        valuetext = f"{value:{item.number_format}}"

                    # Slight realignment of label for centered text option
                    if self.center_text:
                        xshift = 0
                        va = "top" if ivalue == 0 else "bottom"

                    # Draw text
                    txt = ax.text(i+xshift+xoffset,
                                  value,
                                  valuetext,
                                  fontsize=item.fontsize,
                                  color=item.fontcolor,
                                  va=va, ha=ha,
                                  alpha=item.alpha)
                    element[f"box-text{ivalue}"] = txt

            # Draw min / max
            item = self.minmax
            if item.marker != "none":
                x = [i+xoffset]*2
                y = stats.loc[["min", "max"], colname]
                ax.plot(x, y, item.marker,
                        mfc=item.markerfacecolor,
                        mec=item.markeredgecolor,
                        markersize=item.markersize,
                        alpha=item.alpha)

                element["minmax"] = ax.get_lines()[-1]

            # Store element
            self.elements[colname] = element

        # X tick labels
        ax.set_xticks(range(ncols))
        ax.set_xticklabels(stats.columns)

        if self._by is not None:
            ax.set_xlabel(self._by.name)

        # Y axis log scale
        if logscale:
            ax.set_yscale("log", nonpositive="clip")

        # X/Y limits
        w = np.max(boxwidths)
        ax.set_xlim((-w, ncols-1+w))

        ylim = ax.get_ylim()
        if not logscale:
            dy = ylim[1]-ylim[0]
            ylim0 = min(ylim[0], stats.loc[wqq1txt, :].min()-dy*0.02)
            ylim1 = max(ylim[1], stats.loc[wqq2txt, :].max()+dy*0.02)
        else:
            dy = math.log(ylim[1])-math.log(ylim[0])
            ylim0 = ylim[0]
            miniq = stats.loc[wqq1txt, :].min()
            if miniq > 0:
                ylim0 = min(ylim[0], math.exp(math.log(miniq)-dy*0.02))

            ylim1 = ylim[1]
            maxiq = stats.loc[wqq1txt, :].max()
            if maxiq > 0:
                ylim1 = max(ylim[1], math.exp(math.log(maxiq)+dy*0.02))

        ax.set_ylim((ylim0, ylim1))

    def show_count(self, ypos=0.025):
        """ Show the counts """

        ax = self._ax
        if ax is None:
            errmsg = "Boxplot ax is not initialised."\
                     + " Run draw method first."
            raise BoxplotError(errmsg)

        if self.elements is None:
            raise BoxplotError("No elements in the boxplot. Run draw")

        stats = self._stats

        va = "bottom"
        if ypos > 0.5:
            va = "top"

        item = self.count
        if item.show_text:
            # Get y coordinate from ax coordinates
            trans1 = ax.transAxes.transform
            trans2 = ax.transData.inverted().transform
            _, y1 = trans1((0, ypos))
            _, y = trans2((0., y1))

            formatter = "("+item.number_format+")"

            for i, cn in enumerate(stats.columns):
                cnt = formatter % stats.loc["count", cn]
                txt = ax.text(i, y, cnt, fontsize=item.fontsize,
                              color=item.fontcolor,
                              va=va, ha="center")

                if cn in self.elements:
                    self.elements[cn]["count-text"] = txt

        else:
            raise BoxplotError("show_text property for count set to False")

    def set_ylim(self, ylim, hide_offlimit_text=True):
        """ Reset limits of y axis

        Parameters
        -----------
        ylim : tuple
            Y axis limits.
        hide_offlimit_text: bool
            Remove text outside of axis limits.
        """
        if ylim is not None:
            # Make sure ylim is sorted
            ylim = np.sort(ylim)

            # Move text to put it within axis limits
            for elem_name, elem in self.elements.items():
                for item_name, item in elem.items():
                    if re.search("text", item_name):
                        raw = item._y
                        value = max(ylim[0], min(ylim[1], raw))
                        item.set_y(value)

                        # Hide the value if off limits
                        if hide_offlimit_text \
                                and (raw < ylim[0] or raw > ylim[1]):
                            item.set_visible(False)

            # Set axis limits
            self.ax.set_ylim(ylim)

    def set_color(self, name_pattern, color, alpha=0.5):
        """ Set color for selected boxes identied by
        name_pattern.

        Parameters
        -----------
        name_pattern : str
            Pattern selection for box selection.
        color : str
            Color to set
        alpha : float
            Transparency level.
        """
        props = ["color", "linecolor",
                 "edgecolor", "markeredgecolor",
                 "facecolor"]

        for ename, elems in self.elements.items():
            if not re.search(name_pattern, ename):
                continue

            for on, obj in elems.items():
                for prop in props:
                    # Set color to element.
                    # Try all properties with try
                    try:
                        fun = getattr(obj, f"set_{prop}")
                        fun(color)
                    except Exception:
                        pass

                    if re.search("box", on):
                        obj.set_alpha(alpha)
