from unet.model import *
import skimage.io as io
import matplotlib.pyplot as plt
from utils import get_chunked_data, apply_func_by_chunk
import gdal
import numpy as np

# ------ Helpeer function for forest cover loading
def get_data_by_coordinate_np(lats, lons, array, xmin, xres, ymax, yres):
    """Just a helper function"""
    lat_inds = ((lats - ymax) / yres).astype(np.int16)
    lon_inds = ((lons - xmin) / xres).astype(np.int16)
    array = array[lat_inds, lon_inds]
    return array


def load_data(lats, lons, filename):
    data = gdal.Open(filename)
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
    result = get_data_by_coordinate_np(np.array(lats, dtype=np.float64),
                                  np.array(lons, dtype=np.float64),
                                  array,
                                  xmin, xres, ymax, yres)
    return result

# --------------------------------------------------------------------



model = unet()
model.load_weights("unet_windfalls_500_30.hdf5")

# img = io.imread('./data/test/images/70.png')
# plt.figure()
# plt.imshow(img)

# img = np.reshape(img, (1,) + img.shape)
# results = model.predict(img, verbose=1)

# plt.figure()
# plt.imshow(results[0,:,:,0], cmap='gray')

# plt.figure()
# mask = io.imread('./data/test/labels/70.png')
# plt.imshow(mask)
# plt.show()

def predict(img, model=model):
    img = np.reshape(img, (1, ) + img.shape)
    result = model.predict(img, verbose=1)
    return result[0, :, :, 0]



FOREST_THRESHOLD = 75
ind = 0
for window, resolution, img in get_chunked_data(4825000, 5000000,
                                                357200, 495200,
                                                10, 2560 * 5,
                                                [2, 3, 4]):
    lats = np.arange(window[0], window[1], resolution)
    lons = np.arange(window[2], window[3], resolution)
    LA, LO = np.meshgrid(lats, lons)
    # ---------- Getting forest distribution data ---------
    data = load_data(LA, LO, 'forest_cover.tif')
    data = np.flipud(data.T)
    print("Data successfully loaded: ", data.shape)
    print("Image data shape: ", img.shape)
    # -----------------------------------------------------
    data_iterator = apply_func_by_chunk(lambda x: x, data[..., np.newaxis], 256)
    for result, im_chunk, ss in apply_func_by_chunk(predict, img, 256):
        print("iteration index: ", ind)
        fdata, f_chunk = next(data_iterator)
        io.imsave("./images/{}_pred.png".format(ind), result)
        io.imsave("./images/{}_forest.png".format(ind),
                  fdata[..., -1] / fdata[..., -1].max())
        m, n = result.shape
        # result after applying forest distribution mask
        result = result.ravel()
        result[fdata[..., -1].ravel() < FOREST_THRESHOLD] = 0
        result = result.reshape(m, n)
        io.imsave("./images/{}_corr.png".format(ind), result)
        io.imsave("./images/{}_orig.png".format(ind), im_chunk)
        ind += 1

