#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from cartopy import crs


def availability_matrix(df, ax=None, label=True, color={}, bottom=.05, top=.99, **kwargs):
    """Plot a matrix of the times when a given :class:`~pandas.DataFrame` has valid observations. Not sure with what data types it'll still work, but in general 0/False(/nan?) should work for nonexistent times, and 1/count for exisitng ones.

    :param df: DataFrame with time in index and station labels as columns. The columns labels are used to label the rows of the plotted matrix.
    :type df: :class:`~pandas.DataFrame`
    :param ax: :obj:`~matplotlib.axes.Axes.axes` if subplots are used
    :param label: if `False`, plot no row labels
    :type label: :obj:`bool`
    :param color: mapping from color values to row indexes whose labels should be printed in the given color
    :type color: :obj:`dict` {color spec: [row indexes]}

    :Keyword Arguments:
        Same as for :class:`matplotlib.figure.SubplotParams`

    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(kwargs.pop('fig_width', 6), 10 * df.shape[1]/80))
    else:
        fig = ax.figure
    fig.subplots_adjust(bottom=bottom, top=top, **kwargs)
    plt.set_cmap('viridis')
    y = np.arange(df.shape[1] + 1)
    ax.pcolormesh(df.index, y, df.T)
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
    p = ax.scatter(*x.as_matrix().T, marker='o', transform=crs.PlateCarree(), color=color)
    for i, st in x.dropna().iterrows():
        ax.annotate(i, xy=st, xycoords=crs.PlateCarree()._as_mpl_transform(ax), color=p.get_facecolor()[0])
    ax.coastlines()
    ax.gridlines()
    ax.set_extent((-180, 180, -65, -90), crs.PlateCarree())
    return p
