from utils import get_mask_data, generate_train_test, get_data
import matplotlib.pyplot as plt
from skimage import transform
import matplotlib.cm as cm
import matplotlib.lines as lines
import numpy as np

from collections import Counter

# ----------- Coordinate transformation definition ----------

# img, lons, lats = get_mask_data()
# print(img)
# print(img.sum())
# plt.imshow(np.flipud(img))
# plt.show()

NRES = 100
lats = np.arange(4825000, 5000000, NRES)
lons = np.arange(357200, 495200, NRES)
LA, LO = np.meshgrid(lats, lons)
print(LA,LO)
# res = get_data(LA.ravel(), LO.ravel(), [3])
# image = res[0][0].reshape(LA.shape).T
# plt.imshow(np.flipud(image))
# plt.show()

generate_train_test([2,3,4])

