from loader import *
import matplotlib.pyplot as plt
from skimage import transform as tf

from collections import Counter


# resultG = get_arrays(2)
# resultB = get_arrays(3)
# resultR = get_arrays(4)
# plt.imshow((resultG[-1][0]+resultB[-1][0]+resultR[-1][0])/3)
# plt.show()

NRES = 50
lats = np.arange(4825000, 5000000, NRES)
lons = np.arange(357200, 475200, NRES)
LA, LO = np.meshgrid(lats, lons)
result = get_data(LA.ravel(), LO.ravel(), [2])
# plt.imshow(np.flipud(result[0][0].reshape(LA.shape).T))
# plt.show()
# sdf

crop_island = tf.ProjectiveTransform()
latmax = np.max(lats)
latmin = np.min(lats)
lonmax = np.max(lons)
lonmin = np.min(lons)
O = np.array([382212, 4834137])
A = np.array([475187, 4927113])
B = np.array([357294, 4865191])
O = (O - np.array([lonmin, latmin]))/NRES
A = (A - np.array([lonmin, latmin]))/NRES
B = (B - np.array([lonmin, latmin]))/NRES
print("O position:", O)
print("A position:", A)
print("B poisiton:", B)
src = np.array([O.tolist(),
                A.tolist(),
                B.tolist()]
               )
print('Source coordinates', src)
dst = np.array([[0, 0],
                [1000, 0],
                [0, 300]])
crop_island.estimate(src, dst)
print("Estimation copleted....")
img = result[0][0].reshape(LA.shape).T
print('Preparing image data ... ')
print("Wrapping the image: ")
warped = tf.warp(img, crop_island, output_shape=[1000, 300])
print("Wrapping is successfully completed!")
plt.imshow(warped.T)
plt.show()



