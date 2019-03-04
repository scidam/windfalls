from osgeo import gdal, osr
from loader import load_shp_file, get_data
import numpy as np
import os
import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import Point
from conf import *
from skimage import transform, img_as_uint
from skimage.io import imsave
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage import exposure


def array_to_raster(array, lats, lons, asdatafile, fname):
    """Array > Raster
    Save a raster from a C order array.

    :param array: ndarray
    """

    SourceDS = gdal.Open(asdatafile, gdal.GA_ReadOnly)
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
    band = dataset.GetRasterBand(1)
    band.SetNoDataValue(-9999)
    dataset.FlushCache()  # Write to disk.
    return 0

def build_mask_data(minx, maxx, miny, maxy, dx=10, dy=10):
    lons = np.arange(minx, maxx, dx)
    lats = np.arange(miny, maxy, dy)
    LA, LO = np.meshgrid(lats, lons)
    series = gpd.GeoSeries([Point(y, x) for x, y in zip(LA.ravel(), LO.ravel())])
    polygon = load_shp_file()
    return 255 * series.intersects(polygon).values.astype(int).reshape(LA.shape).T, LA, LO


def get_mask_data():
    if os.path.exists(MASKED_PATH):
        return np.load(MASKED_PATH)
    a, b, c = build_mask_data(*TRAIN_WINDOW, MASK_DX_DY, MASK_DX_DY)
    np.save(MASKED_PATH, [a, b, c])
    return a, b, c


def crop_image(img, lats, lons,
               origin=np.array([384745, 4826060]), angle=0.807575,
               length=140 * 10 ** 3,
               longest_size=5000):
    point1 = origin
    point2 = origin + np.array([length * np.cos(angle), length * np.sin(angle)])
    point3 = origin + np.array([length / 3.5 * np.cos(angle + np.pi / 2),
                                length / 3.5 * np.sin(angle + np.pi / 2)])
    point4 = point2 + point3 - point1
    scale = np.linalg.norm(point3 - point1) / np.linalg.norm(point2 - point1)
    minlat = np.min(lats)
    minlon = np.min(lons)
    trans = lambda p: np.array([(p[0] - minlon) / (lons[1] - lons[0]),
                                len(lats) - (p[1] - minlat) / (lats[1] - lats[0])])
    point1 = trans(point1)
    point2 = trans(point2)
    point3 = trans(point3)
    point4 = trans(point4)

    points = (point1, point2, point3, point4)
    destination_y_size = int(longest_size * scale)
    destination_x_size = int(longest_size)

    src = np.array([[0, destination_y_size],
                    [destination_x_size, destination_y_size],
                    [0, 0],
                    [destination_x_size, 0],
                    ])

    dst = np.array([point1,
                    point2,
                    point3,
                    point4])
    crop_island = transform.ProjectiveTransform()
    crop_island.estimate(src, dst)

    warped = transform.warp(np.flipud(img),
                            crop_island,
                            output_shape=[destination_y_size,
                                          destination_x_size])
    return warped, points




def crop_the_island(img, lats, lons):
    """Crops and rotate the island"""

    return crop_image(img, lats, lons)


def generate_train_test(levels):
    train_images = []
    train_labels = []
    mask, lats, lons = get_mask_data()
    sat_data = get_data(lats.ravel(), lons.ravel(), levels)
    sat_layers = [exposure.equalize_adapthist(item[0].reshape(lats.shape).T, clip_limit=0.03) for item in sat_data]
    m, n = mask.shape
    image_num = 0
    for x in range(0, m - SLIDING_WINDOW_SIZE, SLIDING_INCREMENT):
        for y in range(0, n - SLIDING_WINDOW_SIZE, SLIDING_INCREMENT):
            prepared_layers = map(lambda data: data[x : x + SLIDING_WINDOW_SIZE,
                                  y : y + SLIDING_WINDOW_SIZE], sat_layers)
            prepared_mask = mask[x:x + SLIDING_WINDOW_SIZE, y:y + SLIDING_WINDOW_SIZE]
            layer = np.dstack(prepared_layers)

            # check if this image could be added to the train dataset

            train_images.append(np.flipud(layer))
            train_labels.append(np.flipud(prepared_mask))
            # imsave(os.path.join(IMAGES_PATH, str(image_num) + '.png'), np.flipud(layer))
            # imsave(os.path.join(LABELS_PATH, str(image_num) + '.png'), np.flipud(prepared_mask))
            image_num += 1

    # perform filtering procedure according to ZERO_ONE_MIN_FRACTION
    squares = np.array(list(map(lambda x: float(x.sum()) / SLIDING_WINDOW_SIZE / SLIDING_WINDOW_SIZE, train_labels)))
    inds = np.argsort(squares)[::-1]
    retained = list()
    cur = squares[inds[0]]
    for ind in inds[1:]:
        suspect = (cur + squares[ind]) / 2.0
        if suspect > ZERO_ONE_MIN_FRACTION:
            retained.append(ind)
            cur = suspect
    for image_num, ind in enumerate(retained):
        imsave(os.path.join(IMAGES_PATH, str(image_num) + '.png'), train_images[ind])
        imsave(os.path.join(LABELS_PATH, str(image_num) + '.png'), train_labels[ind])


def get_chunked_data(latmin, latmax, lonmin, lonmax, resolution, chunksize, levels):
    for clat in range(latmin, latmax, chunksize):
        for clon in range(lonmin, lonmax, chunksize):
            current_chunk_window = (clat, clat+chunksize, clon, clon+chunksize)
            lats = np.arange(clat, clat + chunksize, resolution)
            lons = np.arange(clon, clon + chunksize, resolution)
            LA, LO = np.meshgrid(lats, lons)
            sat_layers = [exposure.equalize_adapthist(item[0].reshape(LA.shape).T, clip_limit=0.03) for item in get_data(LA.ravel(), LO.ravel(), levels)]
            image_chunk = np.flipud(np.dstack(sat_layers))
            yield current_chunk_window, image_chunk


def apply_func_by_chunk(func, img, chunk_size):
    """chunk_size is given in pixels
    """
    n, m, d = img.shape
    xexceeded, yexceeded = False, False
    for y in range(0, n, chunk_size):
        if y + chunk_size > n:
            yexceeded = True
        for x in range(0, m, chunk_size):
            if x + chunk_size > m:
                xexceeded = True
            if not xexceeded and not yexceeded:
                yield func(img[y:y + chunk_size, x:x + chunk_size, :]), img[y:y + chunk_size, x:x + chunk_size, :]
                continue
            else:
                _ = np.zeros((chunk_size, chunk_size, d), dtype=img.dtype)
                _[:] = np.nan

            if xexceeded and not yexceeded:
                im = img[y:y + chunk_size, x:]
            elif not xexceeded and yexceeded:
                im = img[y:, x:x + chunk_size]
            else:
                im = img[y:, x:]

            _[:im.shape[0], :im.shape[1], :] = im
            yield  func(_), _
        xexceeded = False
        yexceeded = False
