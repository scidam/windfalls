#coding: utf-8

import numpy as np
from functools import lru_cache
import re
import os
from osgeo import gdal
from osgeo import osr
from shapely.ops import cascaded_union
import geopandas as gpd
from shapely.geometry import Point

aux = ['CJ', 'DJ', 'DK']
LARGE_VALUE = 500
COMMON_PATH = './19_09_2017/Scene%s/'
FILE_PAT = "T55T{}_20170919T010641_B{:02}.jp2_Cnv.tif"
scenes = [1,2,3]
layers = range(1, 13)

DATA_PATTERNS = [[os.path.join(COMMON_PATH % s, FILE_PAT.format(aux[s - 1], l)) for s in scenes] for l in layers]

def load_shp_file(filename='winds.shp'):
    data = gpd.read_file(filename)
    polygon = cascaded_union(data.geometry)
    return polygon

def build_mask_data(minx, maxx, miny, maxy, dx=1, dy=1):
    xdata = np.arange(minx, maxx, dx)
    ydata = np.arange(miny, maxy, dy)
    X, Y = np.meshgrid(xdata, ydata)
    series = gpd.GeoSeries([Point(x, y) for x, y in zip(X.ravel(), Y.ravel())])
    polygon = load_shp_file()
    return series.intersects(polygon)


def array_to_raster(array, lats, lons,  fname):
    """Array > Raster
    Save a raster from a C order array.

    :param array: ndarray
    """
    # You need to get those values like you did.

    SourceDS = gdal.Open(DATA_PATTERNS['BIO1']['filename'], gdal.GA_ReadOnly)
    Projection = osr.SpatialReference()
    Projection.ImportFromWkt(SourceDS.GetProjectionRef())
    x_pixels, y_pixels = array.shape
    XPIXEL_SIZE = (lons[1] - lons[0]) / float(x_pixels)
    YPIXEL_SIZE = (lats[1] - lats[0]) / float(y_pixels)
    x_min = np.min(lons)
    y_max = np.max(lats)
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(
        fname,
        y_pixels,
        x_pixels,
        1,
        gdal.GDT_Float32)

    dataset.SetGeoTransform((
        x_min,    # 0
        abs(XPIXEL_SIZE),  # 1
        0,                      # 2
        y_max,    # 3
        0,                      # 4
        -abs(YPIXEL_SIZE)))
    dataset.SetProjection(Projection.ExportToWkt())
    dataset.GetRasterBand(1).WriteArray(array)
    dataset.FlushCache()  # Write to disk.
    return 0


def get_data_by_coordinate_np(lats, lons, array, xmin, xres, ymax, yres):
    lat_inds = ((lats - ymax) / yres).astype(np.int16)
    lon_inds = ((lons - xmin) / xres).astype(np.int16)
    lat_ind_max, lon_ind_max = array.shape
    mask_lat = (lat_inds >= 0) & (lat_inds <= lat_ind_max)
    mask_lon = (lon_inds >= 0) & (lon_inds <= lon_ind_max)
    latitudes = np.arange(ymax, ymax - lat_ind_max * yres, -yres)
    longitudes = np.arange(xmin, xmin + lon_ind_max * xres, xres)
    LA, LO = np.meshgrid(latitudes, longitudes)
    _res = array[lat_inds[mask_lat], lon_inds[mask_lon]]
    _LA = LA[lat_inds[mask_lat], lon_inds[mask_lon]]
    _LO = LO[lat_inds[mask_lat], lon_inds[mask_lon]]
    _res[np.abs(_res) > LARGE_VALUE] = np.nan
    return (array, _LA, _LO)


def get_arrays(level):
    result = list()
    for f in DATA_PATTERNS[level - 1]:
        data = gdal.Open(f)
        print(data.GetGeoTransform())
        if osr.SpatialReference(data.GetProjection()).IsProjected():
            geoinfo = data.GetGeoTransform()
    xmin = geoinfo[0]
    xres = geoinfo[1]
    ymax = geoinfo[3]
    yrot = geoinfo[4]
    xrot = geoinfo[2]
    yres = geoinfo[-1]
    for f in DATA_PATTERNS[level - 1]:
        data = gdal.Open(f)
        if not np.isclose(xrot, 0) or not np.isclose(yrot, 0):
            raise BaseException("xrot and yrot should be 0")
        array = data.ReadAsArray()
        del data
        result.append((array, xmin, xres, ymax, yres))
    return result


def get_data(lats, lons, levels):
    result = []
    for l in levels:
        array, xmin, xres, ymax, yres = get_arrays(l)
        result.append(get_data_by_coordinate_np(np.array(lats, dtype=np.float64),
                                  np.array(lons, dtype=np.float64),
                                  np.array(array, dtype=np.float64),
                                  xmin, xres, ymax, yres))
    return result





