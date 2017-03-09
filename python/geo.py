#!/usr/bin/env python
import gdal


# line drawing algorithm
# https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
def path(s, a):
    # station location in grid pixels
    # g[0], g[3] - top left corner of top left pixel
    x = (st.lon - g[0]) // g[1]
    sy = (st.lat - g[3]) // g[5]
    # correct it to array coordinates (from lower left corner)
    y = ds.RasterYSize - sy - 1
    i = np.arange(ds.RasterXSize)
    j = np.arange(ds.RasterYSize)

    sa = np.sin(a)
    ca = np.cos(a)
    if ca == 0:
        i = np.ones(len(j), dtype=np.int) * x
    if sa == 0:
        j = np.ones(len(i), dtype=np.int) * y
    else:
        ta = abs(np.tan(a))
        if ta < 1:
            j = np.round(np.sign(sa) * ta * i)
            # shift so that line goes through station pixel
            j += y - j[y]
        else:
            i = np.round(np.sign(ca) * abs(ca / sa) * j)
            i += x - i[x]
        return (i + sx).astype(int), (j + sy).astype(int)


ds = gdal.Open('../../data/geo/merged.tif')
g = ds.GetGeoTransform()
b = ds.GetRasterBand(1)
z = b.ReadAsArray()

i, j, d = nearest(lon, lat, st.lon, st.lat)
k, l = path(s, lon[i, j], lat[i, j])
