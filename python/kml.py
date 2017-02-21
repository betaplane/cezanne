from simplekml import Kml, Style
from helpers import nearest
from shapely.geometry import Polygon, Point



def kml(sta, nc=None):
    if nc is not None:
        x = nc.variables['XLONG_M'][0,:,:]
        y = nc.variables['XLAT_M'][0,:,:]
        xc = nc.variables['XLONG_C'][0,:,:]
        yc = nc.variables['XLAT_C'][0,:,:]

    k = Kml()
    for c,s in sta.iterrows():
        lat = s['lat']
        lon = s['lon']
        p = k.newpoint(name = c, coords = [(lon,lat)])
        p.style.iconstyle.icon.href = "http://maps.google.com/mapfiles/kml/paddle/red-circle.png"
        p.style.balloonstyle.text = s['name']
        if nc is not None:
            i,j,d = nearest(x, y, lon, lat)
            coords = [
                (xc[i,j],yc[i,j]),
                (xc[i,j+1],yc[i,j]),
                (xc[i,j+1],yc[i+1,j]),
                (xc[i,j],yc[i+1,j]),
                (xc[i,j],yc[i,j])
            ]
            if Polygon(coords).contains(Point(lon,lat)):
                l = k.newlinestring(coords = [(lon,lat), (x[i,j],y[i,j])])
                r = k.newlinestring(coords=coords)
    return k
