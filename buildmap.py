from unet.model import *
import skimage.io as io
import matplotlib.pyplot as plt
from utils import get_chunked_data, apply_func_by_chunk

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


ind = 0
for chunk, img in get_chunked_data(4825000, 5000000, 357200, 495200, 10,
                                   2560 * 5, [2, 3, 4]):
    for result, im_chunk in apply_func_by_chunk(predict, img, 256):
       io.imsave("{}.png".format(ind), result)
       io.imsave("{}_.png".format(ind), im_chunk)
       ind += 1
