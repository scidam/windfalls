from unet.model import *
import skimage.io as io
import matplotlib.pyplot as plt



model = unet()
model.load_weights("./unet/unet_windfalls.hdf5")

img = io.imread('./data/test/images/55.png')

plt.imshow(img)
plt.show()




img = np.reshape(img, (1,) + img.shape)

results = model.predict(img, verbose=1)



plt.imshow(results[0,:,:,0], cmap='gray')
plt.show()

mask = io.imread('./data/test/labels/55.png')
plt.imshow(mask)
plt.show()