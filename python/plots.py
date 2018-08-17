#!/usr/bin/env python
import os
from importlib import import_module
import numpy as np
import matplotlib.pyplot as plt
from cartopy import crs
from traitlets.config.configurable import Configurable
from traitlets import List
from helpers import stationize


def availability_matrix(df, ax=None, label=True, color={}, bottom=.05, top=.99, **kwargs):
    """Plot a matrix of the times when a given :class:`~pandas.DataFrame` has valid observations. Not sure with what data types it'll still work, but in general 0/False(/nan?) should work for nonexistent times, and 1/count for exisitng ones. Figure size is automatically computed, but can be overridden with the **figsize** or **fig_width** arguments.

    :param df: DataFrame with time in index and station labels as columns. The columns labels are used to label the rows of the plotted matrix. The given DataFrame is attempted to pass through :meth:`python.helpers.stationize` first.
    :type df: :class:`~pandas.DataFrame`
    :param ax: :obj:`~matplotlib.axes.Axes.axes` if subplots are used
    :param label: if `False`, plot no row labels
    :type label: :obj:`bool`
    :param color: mapping from color values to row indexes whose labels should be printed in the given color
    :type color: :obj:`dict` {color spec: [row indexes]}
    :param bottom: equivalent to ``bottom`` keyword in :class:`matplotlib.figure.SubplotParams`
    :param top: equivalent to ``top`` keyword in :class:`matplotlib.figure.SubplotParams`

    :Keyword Arguments:
        Same as for :class:`matplotlib.figure.SubplotParams`, plus:
            * **figsize** - override automatic figure sizing
            * **fig_width** - override only the figure width

    """
    if ax is None:
        figsize = kwargs.pop('figsize', (kwargs.pop('fig_width', 6), 10 * df.shape[1]/80))
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    try: df = stationize(df)
    except: pass
    fig.subplots_adjust(bottom=bottom, top=top, **kwargs)
    plt.set_cmap('viridis')
    y = np.arange(df.shape[1] + 1)
    ax.pcolormesh(df.index, y, df.T, vmin=0., vmax=1.)
    ax.set_yticks(y[1:])
    if label:
        l = ax.set_yticklabels(df.columns)
        for k in l:
            k.set_verticalalignment('bottom')
            k.set_fontsize(8)
        for c, i in color.items():
            for j in i:
                l[j].set_color(c)
    else:
        ax.set_yticklabels([])
    ax.yaxis.set_tick_params(tick1On=False)
    ax.grid()
    ax.invert_yaxis()


def annotated(df, ax=None, color=None):
    """Create a scatterplot on a map where each marker is labeled.

    :param df: DataFrame with longitude and latitude (in that order) as columns and labels in the index
    :type df: :class:`~pandas.DataFrame`
    :param ax: axes instance to use for plotting, if any
    :type ax: :obj:`~matplotlib.axes.Axes.axes`
    :param color: color spec, if any

    """
    ax = plt.gca() if ax is None else ax
    p = ax.scatter(*df.as_matrix().T, marker='o', transform=crs.PlateCarree(), color=color)
    for i, st in df.dropna().iterrows():
        ax.annotate(i, xy=st, xycoords=crs.PlateCarree()._as_mpl_transform(ax), color=p.get_facecolor()[0])
    ax.coastlines()
    ax.gridlines()
    ax.set_extent((-180, 180, -65, -90), crs.PlateCarree())
    return p

def title(fig, title, height=.94):
    ax = fig.add_axes([0, 0, 1, height])
    ax.axis('off')
    ax.set_title(title)
    fig.draw_artist(ax)

def cbar(plot, loc='right', center=False, width=.01, space=.01, lat=-65):
    """Wrapper to attach colorbar on either side of a :class:`~matplotlib.axes.Axes.plot` and to add coastlines and grids.

    :param plot: Plot to attach the colorbar to.
    :type plot: :class:`~matplotlib.axes.Axes.plot`
    :param loc: 'left' or 'right'
    :param center: Whether or not the colors should be centered (divergent).
    :type center: :obj:`bool`
    :param width: Width of the colorbar.
    :param space: Space between edge of plot and colorbar.
    :param lat: latitude circle at which to cut of the plot.

    """
    try:
        ax = plot.ax
    except AttributeError:
        ax = plot.axes
    bb = ax.get_position()
    x = bb.x0 - space - width if loc=='left' else bb.x1 + space
    cax = ax.figure.add_axes([x, bb.y0, width, bb.y1-bb.y0])
    plt.colorbar(plot, cax=cax)
    cax.yaxis.set_ticks_position(loc)
    # ax.coastlines()
    # ax.gridlines()
    # ax.set_extent((-180, 180, -90, lat), crs.PlateCarree())
    if center is not False:
        lim = np.abs(plot.get_clim()).max() if isinstance(center, bool) else center
        plot.set_clim(-lim, lim)

class Coquimbo(Configurable):
    """Add map features for Coquimbo region to a given :class:`~cartopy.mpl.geoaxes.GeoAxes` instance. Usage::

        from cartopy import crs
        coq = Coquimbo()
        ax = plt.axes(projection = crs.PlateCarree())
        coq(ax)

    :Keyword Arguments:
        * **lines_only** - only draw coasline and country border without area fill
        * **colors** - :obj:`iterable` of one or two colors to be used for (coast, border)
    """

    bbox = List([-72.2, -69.8, -32.5, -28.2])
    """configurable bounding box (minx, miny, maxx, maxy) of the region"""

    def __init__(self):
        loader = import_module('traitlets.config.loader')
        super().__init__(
            config = loader.PyFileConfigLoader(
                os.path.expanduser('~/Dropbox/work/config.py')).load_config()
        )
        gshhs = import_module('data.GSHHS')
        self.coast = self.clip(gshhs.GSHHS_Reader('GSHHS_shp/i/GSHHS_i_L1'))
        self.border = self.clip(gshhs.GSHHS_Reader('WDBII_shp/i/WDBII_border_i_L1'))
        self.rivers = self.clip(gshhs.GSHHS_Reader('WDBII_shp/i/WDBII_river_i_L05'))

    def __call__(self, ax, proj=crs.PlateCarree(), lines_only=False, colors=['k']):
        if lines_only:
            ax.add_geometries(self.coast, crs=proj, facecolor='none', edgecolor=colors[0], zorder=10)
            ax.add_geometries(self.border, crs=proj, facecolor='none', edgecolor=colors[-1], linewidth=.5, zorder=10)
        else:
            ax.background_patch.set_color('lightblue')
            ax.add_geometries(self.coast, crs=proj, facecolor='lightgray', edgecolor='k', zorder=0)
            ax.add_geometries(self.rivers, crs=proj, facecolor='none', edgecolor='b', zorder=0)
            ax.add_geometries(self.border, crs=proj, facecolor='none', edgecolor='g', linewidth=.5, zorder=0)
            ax.set_extent(self.bbox, crs=proj)

    def clip(self, reader):
        f = lambda b: np.all(np.r_[b[:2], self.bbox[:2]] <= np.r_[self.bbox[2:], b[2:]])
        return [g for g in reader.geometries() if f(np.array(g.envelope.bounds))]
