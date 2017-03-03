#!/usr/bin/env python
import gdal



# line drawing algorithm
# https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
def path(s,x,y):
	sx = (s['lon']-g[0]-0.5*g[1])//g[1] + (x>s['lon'])
	sy = (s['lat']-g[3]-0.5*g[5])//g[5] + (y<s['lat'])
	dx = abs((g[0]+(sx+0.5)*g[1]-x)//g[1])
	dy = abs((g[3]+(sy+0.5)*g[5]-y)//g[5])
	if dx>dy:
		b = (y-s['lat'])/(x-s['lon'])
		i = np.arange(dx+1)*np.sign(x-s['lon'])
		j = np.round(-b*i)
	else:
		b = (x-s['lon'])/(y-s['lat'])
		j = np.arange(dy+1)*np.sign(s['lat']-y)
		i = np.round(-b*j)
	return (i+sx).astype(int), (j+sy).astype(int)


ds = gdal.Open('../data/DEMS/merged.tif')
g = ds.GetGeoTransform()
b = ds.GetRasterBand(1)
z = b.ReadAsArray()



i,j,d = nearest(lon,lat,s['lon'],s['lat'])
k,l = path(s,lon[i,j],lat[i,j])
