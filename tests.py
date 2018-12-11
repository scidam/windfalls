from loader import *
import matplotlib.pyplot as plt
from skimage import transform as tf
import matplotlib.cm as cm
import matplotlib.lines as lines

from collections import Counter

# resultG = get_arrays(2)
# resultB = get_arrays(3)
# resultR = get_arrays(4)
# plt.imshow((resultG[-1][0]+resultB[-1][0]+resultR[-1][0])/3)
# plt.show()

NRES = 100
lats = np.arange(4825000, 5000000, NRES)
lons = np.arange(357200, 495200, NRES)

#lats = np.arange(4833010, 4833010 + 20000, NRES)
#lons = np.arange(380000, 380000 + 10000, NRES)

LA, LO = np.meshgrid(lats, lons)
result = get_data(LA.ravel(), LO.ravel(), [8])

img = result[0][0]



print("Data loaded ... ")
print(img, lats, lons)
print("=" * 30)


crop_island = tf.ProjectiveTransform()

# ----------- Coordinate transformation definition ----------

angle = 0.807575 # Radians
origin = np.array([384745, 4826060])

length = 140 * 10 ** 3
point1 = origin
point2 = origin + np.array([length * np.cos(angle), length * np.sin(angle)])
point3 = origin + np.array([length/3.5 * np.cos(angle + np.pi / 2), length/3.5 * np.sin(angle + np.pi / 2)])
point4 = point2 + point3 - point1
scale = np.linalg.norm(point3 - point1) / np.linalg.norm(point2 - point1)

# ---------------
maxlat = np.max(lats)
minlat = np.min(lats)
maxlon = np.max(lons)
minlon = np.min(lons)

def trans(p):
    return np.array([(p[0] - minlon) / NRES, len(lats) - (p[1] - minlat) / NRES])

point1 = trans(point1)
point2 = trans(point2)
point3 = trans(point3)
point4 = trans(point4)


plt.imshow(np.flipud(img.reshape(LA.shape).T))
ax = plt.gca()
ax.plot(*point1, 'o')
ax.plot(*point2, 'o')
ax.plot(*point3, 'o')
ax.plot(*point4, 'o')
plt.show()

destination_x_size = 5000
destination_y_size = int(5000 * scale)

src = np.array([[0, destination_y_size],
                [destination_x_size, destination_y_size],
                [0,0],
                [destination_x_size, 0],
               ])

dst = np.array([point1,
                point2,
                point3,
                point4])

print(dst)

crop_island.estimate(src, dst)

# ---------------  Coordinate transformation block end ------------------

img = result[0][0].reshape(LA.shape).T
warped = tf.warp(np.flipud(img),
                 crop_island,
                 output_shape=[destination_y_size, destination_x_size])

print(warped.shape)
plt.imshow(warped, cmap=cm.hsv)
plt.show()


