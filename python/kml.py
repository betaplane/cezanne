from simplekml import Kml, Style
from helpers import nearest
from shapely.geometry import Polygon, Point



def kml(name, lon, lat, code=None, nc=None):
    if nc is not None:
        x = nc.variables['XLONG_M'][0,:,:]
        y = nc.variables['XLAT_M'][0,:,:]
        xc = nc.variables['XLONG_C'][0,:,:]
        yc = nc.variables['XLAT_C'][0,:,:]

    k = Kml()
    z = zip(name, lon, lat) if code is None else zip(name, lon, lat, code)
    for s in z:
        p = k.newpoint(name = s[3] if len(s)==4 else s[0], coords = [s[1:3]])
        p.style.iconstyle.icon.href = "http://maps.google.com/mapfiles/kml/paddle/red-circle.png"
        p.style.balloonstyle.text = s[0]
        if nc is not None:
            i,j,d = nearest(x, y, s[1], s[2])
            coords = [
                (xc[i,j],yc[i,j]),
                (xc[i,j+1],yc[i,j]),
                (xc[i,j+1],yc[i+1,j]),
                (xc[i,j],yc[i+1,j]),
                (xc[i,j],yc[i,j])
            ]
            if Polygon(coords).contains(Point(*s[1:3])):
                l = k.newlinestring(coords = [s[1:3], (x[i, j], y[i, j])])
                r = k.newlinestring(coords=coords)
    return k
