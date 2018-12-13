#coding: utf-8

import numpy as np
import pandas as pd
from functools import lru_cache
import re
import os
from osgeo import gdal
from osgeo import osr
from shapely.ops import cascaded_union, unary_union
import geopandas as gpd
from conf import DATA_PATTERNS



def load_shp_file(filename='./winds/data-polygon.shp'):
    """Loads shape file

    This function is used to generate masked array that represents
    regions of interest.

    NOTE: cascaded_union isn't working due to some geos-lib issues
    (use unary_union instead)
    """
    data = gpd.read_file(filename)
    gg = data.geometry.iloc[0]
    if not gg.is_valid:
        gg = gg.buffer(1)
    for g in data.geometry.iloc[1:]:
        if not g.is_valid:
            _g = g.buffer(1)
        else:
            _g = g
        gg = unary_union([gg, _g])
    polygon = gg
    return polygon

def get_data_by_coordinate_np(lats, lons, array, xmin, xres, ymax, yres):
    lat_inds = ((lats - ymax) / yres).astype(np.int16)
    lon_inds = ((lons - xmin) / xres).astype(np.int16)
    lat_ind_max, lon_ind_max = array.shape
    lat_ind_max -= 1
    lon_ind_max -= 1
    mask_lat = (lat_inds >= 0) & (lat_inds <= lat_ind_max)
    mask_lon = (lon_inds >= 0) & (lon_inds <= lon_ind_max)
    full_mask = mask_lat & mask_lon
    _res = array[lat_inds[full_mask], lon_inds[full_mask]]
    num = len(lats[~full_mask])
    values = np.empty(num)
    values[:] = np.nan
    return (np.hstack([_res, values]),
            np.hstack([lats[full_mask], lats[~full_mask]]),
            np.hstack([lons[full_mask], lons[~full_mask]]))


def get_arrays(level):
    result = list()
    for f in DATA_PATTERNS[level - 1]:
        data = gdal.Open(f)
        geoinfo = data.GetGeoTransform()
        xmin = geoinfo[0]
        xres = geoinfo[1]
        ymax = geoinfo[3]
        yrot = geoinfo[4]
        xrot = geoinfo[2]
        yres = geoinfo[-1]
        if not np.isclose(xrot, 0) or not np.isclose(yrot, 0):
            raise BaseException("xrot and yrot should be 0")
        array = data.ReadAsArray()
        del data
        result.append((array, xmin, xres, ymax, yres))
    return result


def load_list_all_info():
    """Auxiliary function: it doesn't used in the computational process

    """

    for f in sum(DATA_PATTERNS, []):
        data = gdal.Open(f)
        geoinfo = data.GetGeoTransform()
        print(f, geoinfo)



def get_data(lats, lons, levels):
    result = []
    for l in levels:
        intermediate = []
        for array, xmin, xres, ymax, yres in get_arrays(l):
            intermediate.append(get_data_by_coordinate_np(np.array(lats, dtype=np.float64),
                                  np.array(lons, dtype=np.float64),
                                  np.array(array, dtype=np.float64),
                                  xmin, xres, ymax, yres))
        _a, _b, _c = np.array([]), np.array([]), np.array([])
        for a, b, c in intermediate:
            _a = np.append(_a, a)
            _b = np.append(_b, b)
            _c = np.append(_c, c)
        aux = pd.DataFrame({'data': _a, 'lat': _b, 'lon': _c})
        aux_data = aux.loc[~pd.isnull(aux.data), :]
        aux_nodata = aux.loc[pd.isnull(aux.data), :]
        aux_data = aux_data.drop_duplicates(['lat', 'lon'])
        aux_nodata = aux_nodata.drop_duplicates(['lat', 'lon'])
        aux = pd.concat([aux_data, aux_nodata], axis=0)
        aux.drop_duplicates(['lat', 'lon'], inplace=True)
        aux.sort_values(by=['lon', 'lat'], inplace=True)
        result.append([aux.data.values, aux.lat.values, aux.lon.values])
    return result






