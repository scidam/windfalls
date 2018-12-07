from loader import *
import matplotlib.pyplot as plt

# resultG = get_arrays(2)
# resultB = get_arrays(3)
# resultR = get_arrays(4)
# plt.imshow((resultG[-1][0]+resultB[-1][0]+resultR[-1][0])/3)
# plt.show()


lats = np.arange(4911770, 4911770+300, 60)
lons = np.arange(450405, 450405+200, 60)
LA, LO = np.meshgrid(lats, lons)
print(LA.shape, LO.shape)

result = get_data(LA.ravel(), LO.ravel(), [2,3])
print(result[0][0].reshape(LA.shape))
plt.imshow(result[0][0].reshape(LA.shape).T, origin='lower')
plt.show()