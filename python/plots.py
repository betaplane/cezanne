#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker, gridspec as gs
from cartopy import crs
from traitlets.config.configurable import Configurable
from traitlets.config.loader import PyFileConfigLoader, ConfigFileNotFound
from traitlets import List, Unicode
from importlib import import_module
from functools import singledispatch
from helpers import config


def availability_matrix(df, ax=None, label=True, color={}, bottom=.05, top=.99, **kwargs):
    """Plot a matrix of the times when a given :class:`~pandas.DataFrame` has valid observations. Not sure with what data types it'll still work, but in general 0/False(/nan?) should work for nonexistent times, and 1/count for exisitng ones. Figure size is automatically computed, but can be overridden with the **figsize** or **fig_width** arguments.

    :param df: DataFrame with time in index and station labels as columns. The columns labels are used to label the rows of the plotted matrix. The given DataFrame is attempted to pass through :meth:`python.helpers.stationize` first.
    :type df: :class:`~pandas.DataFrame`
    :param ax: :obj:`~matplotlib.axes.Axes.axes` if subplots are used
    :param label: if `False`, plot no row labels
    :type label: :obj:`bool`
    :param color: mapping from station name (DataFrame column name) to color in which the corresponding label should be printed
    :type color: :obj:`dict` {row label: color spec}
    :param bottom: equivalent to ``bottom`` keyword in :class:`matplotlib.figure.SubplotParams`
    :param top: equivalent to ``top`` keyword in :class:`matplotlib.figure.SubplotParams`

    :Keyword Arguments:
        Same as for :class:`matplotlib.figure.SubplotParams`, plus:
            * **figsize** - override automatic figure sizing
            * **fig_width** - override only the figure width
            * **grid_color** - color spec for the grid over the matrix

    """
    if ax is None:
        figsize = kwargs.pop('figsize', (kwargs.pop('fig_width', 6), 10 * df.shape[1]/80))
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    try:
        hh = import_module('helpers')
        df = hh.stationize(df)
    except: pass
    grid_color = kwargs.pop('grid_color', plt.rcParams['grid.color'])
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
        for i, c in enumerate(df.columns):
            l[i].set_color(color[c])
    else:
        ax.set_yticklabels([])
    ax.yaxis.set_tick_params(tick1On=False)
    ax.grid(color=grid_color)
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

def add_axes(subplot_spec):
    ax = plt.gcf().add_subplot(subplot_spec)
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax

@singledispatch
def bottom_label(obj, label, size=12, **kwargs):
    obj.set_xlabel(label, size=size, **kwargs)
    return obj

@bottom_label.register(gs.SubplotSpec)
def _(obj, *args, **kwargs):
    return bottom_label(add_axes(obj), *args, **kwargs)

@singledispatch
def row_label(obj, label, rotation=0, size=12, labelpad=40, **kwargs):
    obj.set_ylabel(label, rotation=rotation, size=size, labelpad=labelpad, **kwargs)
    return obj

@row_label.register(gs.SubplotSpec)
def _(obj, *args, **kwargs):
    return row_label(add_axes(obj), *args, **kwargs)

def cbar(plot, loc='right', cax=None, center=False, width=.01, space=.01, label=None, label_kw={}):
    """Wrapper to attach colorbar on either side of a :class:`~matplotlib.axes.Axes.plot` and to add coastlines and grids.

    :param plot: Plot to attach the colorbar to.
    :type plot: :class:`~matplotlib.axes.Axes.plot`
    :param loc: 'left' or 'right'
    :param center: Whether or not the colors should be centered at 0 (divergent).
    :type center: :obj:`bool`
    :param width: Width of the colorbar.
    :param space: Space between edge of plot and colorbar.
    :param label: Label for the colorbar (units) - currently placed inside the colorbar.
    :param label_kw: :obj:`dict` of additional keyword arguments for the label text.

    """
    if cax is None:
        try:
            ax = plot.ax
        except AttributeError:
            ax = plot.axes
        bb = ax.get_position()
        x = bb.x0 - space - width if loc=='left' else bb.x1 + space
        cax = ax.figure.add_axes([x, bb.y0, width, bb.y1-bb.y0])
    elif isinstance(cax, gs.SubplotSpec):
        cax = plot.figure.add_subplot(cax)

    cb = plt.colorbar(plot, cax=cax)
    cax.yaxis.set_ticks_position(loc)
    if center is not False:
        lim = np.abs(plot.get_clim()).max() if isinstance(center, bool) else center
        plot.set_clim(-lim, lim)

    if label is not None:
        cax.text(.5, .95, label, ha='center', va='top', size=12, fontweight='bold', usetex=True, **label_kw)

    return cb

class Coquimbo(config.Coquimbo):
    """Add map features for Coquimbo region to a given :class:`~cartopy.mpl.geoaxes.GeoAxes` instance. Usage::

        from cartopy import crs
        coq = Coquimbo()
        ax = plt.axes(projection = crs.PlateCarree())
        coq(ax)

    :Keyword Arguments:
        * **lines_only** - only draw coasline and country border without area fill
        * **colors** - :obj:`iterable` of one or two colors to be used for (coast, border)

    """
    def __init__(self):
        gshhs = import_module('data.GSHHS')
        self.coast = self.clip(gshhs.GSHHS('GSHHS_shp/i/GSHHS_i_L1'))
        self.border = self.clip(gshhs.GSHHS('WDBII_shp/i/WDBII_border_i_L1'))
        self.rivers = self.clip(gshhs.GSHHS('WDBII_shp/i/WDBII_river_i_L05'))

    def __call__(self, ax, proj=crs.PlateCarree(), lines_only=False, colors='w', transparent=False):
        if isinstance(colors, str):
            colors = {'coast': colors, 'border': colors, 'outline': colors}
        if lines_only:
            ax.add_geometries(self.coast, crs=proj, facecolor='none', edgecolor=colors['coast'], zorder=10)
            ax.add_geometries(self.border, crs=proj, facecolor='none', edgecolor=colors['border'], linewidth=.5, zorder=10)
            if transparent:
                ax.background_patch.set_color('none')
                ax.outline_patch.set_edgecolor(colors['outline'])
        else:
            ax.background_patch.set_color('lightblue')
            ax.add_geometries(self.coast, crs=proj, facecolor='lightgray', edgecolor='k', zorder=0)
            ax.add_geometries(self.rivers, crs=proj, facecolor='none', edgecolor='b', zorder=0)
            ax.add_geometries(self.border, crs=proj, facecolor='none', edgecolor='g', linewidth=.5, zorder=0)
        ax.set_extent(self.bbox, crs=proj)

    def clip(self, reader):
        f = lambda b: np.all(np.r_[b[:2], self.bbox[:2]] <= np.r_[self.bbox[2:], b[2:]])
        return [g for g in reader.geometries() if f(np.array(g.envelope.bounds))]

    def plotrow(self, df, subplot_spec=gs.GridSpec(1, 1)[0], subplot_kw={}, cbar_kw={}, **kwargs):
        """Plot the columns of :class:`~pandas.DataFrame` ``df`` as a row of Coquimbo-area plots.

        :Keyword Arguments:
            * **subplot_spec** - A :class:`matplotlib.gridspec.SubplotSpec` (e.g. a row from a :class:`matplotlib.gridspec.GridSpec`) in which to embed the generated plot, or ``None``
            * **lonlat** - A :class:`numpy.ndarray` containing longitues, latitudes as columns and stations corresponding to the passed DataFrame as rows. If none is passed, the default 'stations' DataFrame is loaded and indexed according to the index of the passed DataFrame.
            * **vmin** - see :func:`matplotlib.pyplot.scatter`
            * **vmax** - see :func:`matplotlib.pyplot.scatter`
            * **title** - (``True``/``False``) plot titles (taken from column names of input DataFrame).
            * **xlabels** - (``True``/``False``) plot xlabels
            * **ylabels** - (``True``/``False``) plot ylabels
            * **cbar** - Colorbar location (see :func:`matplotlib.pyplot.colorbar`) or ``None`` if none is desired (default 'right').
            * **cbar_kw** - keywords for :func:`cbar`
            * **subplot_kw** - keywords for the constructor of :class:`~matplotlib.gridspec.GridSpecFromSubplotSpec`
            * **norm** - a :class:`~matplotlib.cm.ScalarMappable` (optional, e.g. for categorical data)

        """
        try:
            lonlat = kwargs['lonlat']
        except KeyError:
            sta = pd.read_hdf(config.Meta.file_name, 'stations').loc[df.index]
            lonlat = sta[['lon', 'lat']].astype(float).as_matrix().T
        vmin = kwargs.get('vmin', np.nanmin(df))
        vmax = kwargs.get('vmax', np.nanmax(df))
        geom = subplot_spec.get_geometry()
        g = gs.GridSpecFromSubplotSpec(1, df.shape[1], subplot_spec=subplot_spec, **subplot_kw)
        pl, gls  = [], []
        for i, (n, c) in enumerate(df.iteritems()):
            if c.isnull().all(): continue
            ax = plt.gcf().add_subplot(g[0, i], projection=crs.PlateCarree())
            if 'norm' in kwargs:
                pl.append(ax.scatter(*lonlat, c=kwargs['norm'].to_rgba(c), transform=crs.PlateCarree()))
            else:
                pl.append(ax.scatter(*lonlat, c=c, vmin=vmin, vmax=vmax, transform=crs.PlateCarree()))
            self(ax)
            ax.outline_patch.set_edgecolor('w')
            gl = ax.gridlines(linestyle='--', color='w', draw_labels=True)
            gl.xlocator = ticker.FixedLocator(range(-73, -68))
            gl.ylocator = ticker.FixedLocator(range(-33, -27))
            gl.xlabels_top = False
            gl.ylabels_right = False
            if kwargs.get('title', True) and ((geom[3] is None) or (geom[3] < geom[1])):
                ax.set_title(n)
            if geom[3] is not None:
                if not kwargs.get('xlabels', True) or (geom[3] < (geom[0] - 1) * geom[1]):
                    gl.xlabels_bottom = False
            if not kwargs.get('ylabels', True) or (i > 0):
                gl.ylabels_left = False
            gls.append(gl)
        cb = kwargs.get('cbar', 'right')
        if (cb is not None):
            cbar_kw.setdefault('space', 0.02)
            cbar_kw.setdefault('width', 0.02)
            cbar(pl[{'left': 0, 'right':-1}[cb]], loc=cb, **cbar_kw)
        return pl, gls

def axesColor(ax, color):
    for s in ['bottom', 'top', 'left', 'right']:
        ax.spines[s].set_color(color)

def transparent(obj, **colors):
    ax = obj.axes # works even if obj is an axes instance
    ax.background_patch.set_color('none')
    ax.outline_patch.set_edgecolor(colors.get('outline', 'w'))
    return obj
