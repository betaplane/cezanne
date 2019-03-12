import pandas as pd
from glob import glob
from importlib import import_module
import os

class Reader(object):

    names = ['name', 'description', 'units', 'range']

    # Pre-IceBridge ATM L2 Icessn Elevation, Slope, and Roughness, Version 1
    # https://nsidc.org/data/BLATM2/versions/1
    BLATM2 = {'cols': [
        ['SOD', 'Time at which the aircraft passed the mid-point of the block.', 'Seconds of the day in GPS time. As of 01 January 2009 GPS time = UTC + 15 seconds.', '0, 86400'],
        ['lat', 'Latitude of the center of the block.', 'Degrees', '-90, 90'],
        ['lon', 'East longitude of the center of the block.', 'Degrees', '0, 360'],
        ['elev', 'Height above WGS84 ellipsoid of the center of the block.', 'Meters', '-100.0, 10000.0'],
        ['SN_slope', 'South to North slope of the block.', 'Dimensionless', 'real'],
        ['WE_slope', 'West to East slope of the block.', 'Dimensionless', 'real'],
        ['rms', 'RMS fit of the ATM data to the plane.', 'Centimeters', '> 0.0'],
        ['n_used', 'Number of points used in estimating the plane parameters.', 'Count', '> 0.0'],
        ['n_rem', 'Number of points used in estimating the plane parameters.', 'Count', '> 0.0'],
        ['dist', 'Distance of the center of the block from the centerline of the aircraft trajectory (starboard = positive, port = negative).', 'Meters', 'real'],
        ['id', 'Track identifier (numbered 1...n, starboard to port, and 0 = nadir).', 'Number', 'int']
    ],
              'kwargs': {'delim_whitespace': True, 'header': None}, 'filename_date': 1,
              'base_date': lambda S: pd.Timestamp.strptime(S[1], '%y%m%d'),
              'index_cols': ['SOD'],
              'index_func': lambda T: pd.Timedelta(seconds=float(T[0]))
    }

    # IceBridge KT19 IR Surface Temperature, Version 1
    # https://nsidc.org/data/IAKST1B/versions/1
    IAKST1B = {'cols': [
        ['year', 'Year of measurement', 'Years'],
        ['DOY', 'Day of year of measurement', 'Days'],
        ['SOD', 'Seconds of day of measurement (UTC)', 'Seconds'],
        ['lat', 'Latitude of GPS antenna', 'Decimal degrees'],
        ['lon', 'Longitude of GPS antenna', 'Decimal degrees'],
        ['alt', 'Height of GPS antenna above WGS84 ellipsoid', 'Meters'],
        ['T', 'Surface temperature measured by the KT19', 'Degrees Celsius'],
        ['IT', 'KT19 instrument internal temperature', 'Degrees Celsius']
    ],
               'kwargs': {'skiprows': 11, 'header': None},
               'index_cols': ['year', 'DOY', 'SOD'],
               'index_func': lambda T: pd.Timestamp(year=int(T[0]), month=1, day=1) + pd.Timedelta(days=int(T[1])-1, seconds=float(T[2]))
    }

    # IceBridge ATM L2 Icessn Elevation, Slope, and Roughness, Version 2
    # https://nsidc.org/data/ILATM2/versions/2
    ILATM2 = {'cols': [
        ['SOD', 'Time at which the aircraft passed the mid-point of the block.', 'Seconds of the day in UTC', '0, 86400'],
        ['lat', 'Latitude of the center of the block.', 'Degrees', '-90.0, 90.0'],
        ['lon', 'East longitude of the center of the block.', 'Degrees', '0.0, 360.0'],
        ['elev', 'Height above WGS84 ellipsoid of the center of the block.', 'Meters', '-100.0, 10000.0'],
        ['SN_slope', 'South to North slope of the block.', 'Dimensionless', 'real'],
        ['WE_slope', 'West to East slope of the block.', 'Dimensionless', 'real'],
        ['rms', 'RMS fit of the ATM data to the plane.', 'Centimeters', '> 0.0'],
        ['n_used', 'Number of points used in estimating the plane parameters.', 'Count', '> 0'],
        ['n_rem', 'Number of points removed in estimating the plane parameters.', 'Count', '> 0'],
        ['dist', 'Distance of the center of the block from the centerline of the aircraft trajectory (starboard = positive, port = negative).', 'Meters', 'real'],
        ['id', 'Track identifier (numbered 1...n, starboard to port, and 0 = nadir).', 'Number', 'int']
    ],
              'kwargs': {'skiprows': 10, 'header': None}, 
              'base_date': lambda S: pd.Timestamp.strptime(S[1], '%Y%m%d'),
              'index_cols': ['SOD'],
              'index_func': lambda T: pd.Timedelta(seconds=float(T[0]))
    }

    # IceBridge Riegl Laser Altimeter L2 Geolocated Surface Elevation Triplets, Version 1
    # https://nsidc.org/data/ILUTP2/versions/1
    ILUTP2 = {'cols': [
        ['year', 'Year', 'UTC'],
        ['DOY', 'Day of Year', 'UTC'],
        ['SOD', 'Second of day', 'UTC'],
        ['lon', 'Longitude Angle, WGS-84', 'Degrees'],
        ['lat', 'Latitude Angle, WGS-84', 'Degrees'],
        ['elev', 'Laser Derived Surface Elevation (WGS-84, ITRF2005)', 'Meters']
    ],
              'kwargs': {'delim_whitespace': True, 'skiprows': 44, 'header': None},
              'index_cols': ['year', 'DOY', 'SOD'],
              'index_func': lambda T: pd.Timestamp(year=int(T[0]), month=1, day=1) + pd.Timedelta(days=int(T[1])-1, seconds=float(T[2]))
    }

    # IceBridge LVIS L2 Geolocated Surface Elevation Product, Version 1
    # https://nsidc.org/data/ILVIS2/versions/1
    ILVIS2 = {'cols': [
        ['LVIS_LFID', 'LVIS file identification, including date and time of collection and file number. Values three through seven in the first field represent the Modified Julian Date of the data collection.', 'n/a'],
        ['SHOTNUMBER', 'Laser shot assigned during collection', 'n/a'],
        ['SOD', 'UTC decimal seconds of the day', 'Seconds'],
        ['lon', 'Centroid longitude of the corresponding LVIS Level-1B waveform', 'Degrees east'],
        ['lat', 'Centroid latitude of the corresponding LVIS Level-1B waveform', 'Degrees north'],
        ['elev', 'Centroid elevation of the corresponding LVIS Level-1B waveform', 'Meters'],
        ['lon_l', 'Longitude of the center of the lowest mode in the waveform',	'Degrees east'],
        ['lat_l', 'Latitude of the center of the lowest mode in the waveform', 'Degrees north'],
        ['elev_l', 'Elevation of the center of the lowest mode in the waveform', 'Meters'],
        ['lon_h', 'Longitude of the center of the highest mode in the waveform', 'Degrees east'],
        ['lat_h', 'Latitude of the center of the highest mode in the waveform', 'Degrees north'],
        ['elev_h', 'Elevation of the center of the highest mode in the waveform', 'Meters']
    ],
              'kwargs': {'delim_whitespace': True, 'skiprows': 2, 'header': None},
              'base_date': lambda S: pd.Timestamp.strptime(''.join(S[1:3])[2:], '%Y%m%d'),
              'index_cols': ['SOD'],
              'index_func': lambda T: pd.Timedelta(seconds=float(T[0]))
    }

    # IceBridge HiCARS 2 L2 Geolocated Ice Thickness, Version 1
    # https://nsidc.org/data/IR2HI2/versions/1
    IR2HI2 = {'cols': [
        ['year', 'Year', 'UTC'],
        ['DOY', 'Day Of Year', 'UTC'],
        ['SOD', 'Second Of Day', 'UTC'],
        ['lon', 'Longitude', 'Decimal degrees, WGS-84'],
        ['lat',	'Latitude', 'Decimal degrees, WGS-84'],
        ['THK', 'Radar Derived Ice Thickness using dielectric of ice of 1.78 and no firn correction', 'Meters'],
        ['SRF_RNG', 'Radar Derived Surface Range', 'Meters'],
        ['bed', 'Radar Derived Bed Elevation', 'Meters, WGS-84'],
        ['sfc',	'Radar Derived Surface Elevation', 'Meters, WGS-84'],
        ['PARTIAL_BED_REFLECT', 'Bed reflection coefficient @ 60 MHz', 'Decibels with reference to perfect reflector; no ice loss accounting'],
        ['SRF_REFLECT', 'Surface reflection coefficient @ 60 MHz', 'Decibels with reference to perfect reflector'],
        ['AIRCRAFT_ROLL', 'Roll, right wing down positive', 'Degrees'],
    ],
              'kwargs': {'delim_whitespace': True, 'skiprows': 65, 'header': None},
              'index_cols': ['year', 'DOY', 'SOD'],
              'index_func': lambda T: pd.Timestamp(year=int(T[0]), month=1, day=1) + pd.Timedelta(days=int(T[1])-1, seconds=float(T[2]))
    }

    # IceBridge MCoRDS L2 Ice Thickness, Version 1
    # https://nsidc.org/data/IRMCR2/versions/1
    IRMCR2 = {'cols': [
        ['lat', 'Latitude', 'Degrees North'],
        ['lon', 'Longitude', 'Degrees East'],
        ['SOD', 'UTC Time', 'Seconds of day'],
        ['THK', 'Ice Thickness: Bottom minus Surface. Constant dielectric of 3.15 (no firn) is assumed for converting propagation delay into range. -9999 indicates no thickness available.', 'Meters'],
        ['elev', 'Elevation referenced to WGS-84 Ellipsoid.', 'Meters'],
        ['frame', 'Fixed length numeric field (YYYYMMDDSSFFF). YYYY = year, MM = month, DD = day, SS = segment, FFF = frame.', 'N/A'],
        ['sfc_range', 'Range to Ice Surface. Actual surface height is Elevation minus this number.', 'Meters'],
        ['Bottom', 'Range to Ice Bottom. Actual ice bottom height is Elevation minus this number. Constant dielectric of 3.15 (no firn) is assumed for converting propagation delay into range. -9999 indicates no thickness available.', 'Meters'],
        ['Quality', '1: High confidence pick; 2: Medium confidence pick; 3: Low confidence pick', '']
    ],
              'kwargs': {'skiprows': 1, 'header': None},
              'base_date': lambda S: pd.Timestamp.strptime(S[1], '%Y%m%d'),
              'index_cols': ['SOD'],
              'index_func': lambda T: pd.Timedelta(seconds=float(T[0]))
    }

    def __init__(self, pattern):
        self.dfs = dict([(os.path.splitext(os.path.split(f)[-1])[0], self.read_file(f)) for f in glob(pattern)])

    def read_file(self, filename):
        print('reading file {}'.format(filename))
        a = os.path.split(filename)[-1].split('_')
        params = getattr(self, a[0])
        df = pd.read_csv(filename, **params['kwargs'])
        cols = params['cols']
        df.columns = pd.MultiIndex.from_tuples(cols,  names=self.names[:len(cols[0])])
        return self.make_index(df, params, a)

    def make_index(self, df, params, a):
        cols = params['index_cols']
        index = [params['index_func'](i) for i in df[cols].as_matrix()]
        if 'base_date' in params:
            df.index = pd.np.array(index) + params['base_date'](a)
        else:
            df.index = index
        return df.drop(cols, 1)

    def to_shapefile(self, filename):
        geom = import_module('shapely.geometry')
        ogr = import_module('osgeo.ogr')
        ds = ogr.GetDriverByName('Esri Shapefile').CreateDataSource(filename)
        layer = ds.CreateLayer('', None, ogr.wkbLineString)
        layer.CreateField(ogr.FieldDefn('filename', ogr.OFTString))

        for k, df in self.dfs.items():
            lon, lat = df[['lon', 'lat']].as_matrix().T
            if (max(lon) > 180):
                lon[lon > 180] = lon[lon > 180] - 360
            lstr = geom.LineString(zip(lon, lat))
            feat = ogr.Feature(layer.GetLayerDefn())
            feat.SetField('filename', k)
            feat.SetGeometry(ogr.CreateGeometryFromWkb(lstr.wkb))
            layer.CreateFeature(feat)
