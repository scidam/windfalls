import os

# ------------------ Data loading variables ----------------

AUX = ['CJ', 'DJ', 'DK']
LARGE_VALUE = 500
COMMON_PATH = './data/imgs/'
FILE_PAT = "T55T{}_20170919T010641_B{:02}.jp2"
SCENES = [1,2,3]
LAYERS = range(1, 13)

DATA_PATTERNS = [[os.path.join(COMMON_PATH, FILE_PAT.format(AUX[s - 1], l)) for s in SCENES] for l in LAYERS]
# ------------------------------------------------------------



# ------------- Building train and test datasets -------------
TRAIN_WINDOW = [396507, 409668, 4861753, 4874657]
             #  MIN_LON, MAX_LON, MIN_LAT, MAX_LAT



COMM_TRAIN_PATH = './data/train'
IMAGES = 'images'
LABELS = 'labels'

MASK_DX_DY = 10
MASKED_PATH = './masked/data.npy'

LABELS_PATH = os.path.join(COMM_TRAIN_PATH, LABELS)
IMAGES_PATH = os.path.join(COMM_TRAIN_PATH, IMAGES)

SLIDING_WINDOW_SIZE = 256
SLIDING_INCREMENT = 20 # the same value fo lats and lons...


# ------------------------------------------------------------


# ALl the window of interest
#NRES = 100
#lats = np.arange(4825000, 5000000, NRES)
#lons = np.arange(357200, 495200, NRES)



# ------------ Old data... probably invalid
# aux = ['CJ', 'DJ', 'DK']
# LARGE_VALUE = 500
# COMMON_PATH = './Sentinel2/19.09.2017/Scene%s/'
# FILE_PAT = "T55T{}_20170919T010641_B{:02}.jp2_Cnv.tif"
# scenes = [1,2,3]
# layers = range(1, 13)

