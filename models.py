import numpy as np
from sklearn.ensemble import RandomForestClassifier
from skimage import exposure
from sklearn.feature_selection import RFECV
from conf import RF_CONF, SAT_LAYERS, RF_CONF
from utils import get_mask_data, get_data
from sklearn.model_selection import cross_val_score

rf_clf = RandomForestClassifier(**RF_CONF)

mask, lats, lons = get_mask_data()
mask = (mask / 255).astype(int)
print("Mask data is loaded.")

if any(SAT_LAYERS):
    sat_data = get_data(lats.ravel(), lons.ravel(), SAT_LAYERS)
else:
    sat_data = get_data(lats.ravel(), lons.ravel(), list(range(1, 13)))

print("Sattelite data is loaded.")

print("Preparing train dataset...")
X = np.array([exposure.equalize_adapthist(item[0].reshape(lats.shape).T,
                                          clip_limit=0.03).ravel() for item\
              in sat_data]).T
Y = mask.ravel()
print("Train dataset is formed.")

# if layers aresn't defined apply recursive feature elimination procedure to the full set of features/layers
features_mask = None
if not any(SAT_LAYERS):
    print("Performing recursive feature ellimination...")
    rfecv_clf = RFECV(rf_clf, step=1, min_features_to_select=2, scoring='f1',
                      n_jobs=3, verbose=1)
    rfecv_clf.fit(X, Y)
    rf_clf = rfecv_clf.estimator_
    print("Selected features are: ", rf_clf.support_)
    features_mask = rf_clf.support_
else:
    print("Training the classifier... ")
    rf_clf.fit(X, Y)

if features_mask is None:
    scores = cross_val_score(rf_clf, X, Y, cv=5)
else:
    scores = cross_val_score(rf_clf, rfecv_clf.transform(X), Y, cv=5)

print("Score estimations are: ", scores)




