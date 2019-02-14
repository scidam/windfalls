
from utils import generate_train_test, get_mask_data, get_data, exposure
import logging
import numpy as np
import itertools

# import scikit-learn modules here
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

logging.basicConfig(filename='result.log', filemode='w', level=logging.INFO)
logging.info("=" * 50)
models = [LogisticRegression(),
          GaussianNB(),
          RandomForestClassifier(n_estimators=100, random_state=10),
          AdaBoostClassifier(DecisionTreeClassifier(max_depth=7), n_estimators=100, random_state=10)
          ] # a list om ml models to be tested

mask, lats, lons = get_mask_data()
y = mask.ravel()


logging.info("Models to be evaluated: " + str(models))
logging.info("=" * 80)
for triple in tqdm(list(itertools.combinations(range(1, 13), 3))):
    logging.info("-" * 60)
    logging.info("Checking the triple: " + str(triple))
    sattelite_data = get_data(lats.ravel(), lons.ravel(), triple)
    sat_layers = [exposure.equalize_adapthist(item[0].reshape(lats.shape).T, clip_limit=0.03) for item in sattelite_data]
    prepared_layers = map(lambda x: x.ravel(), sat_layers)
    X = np.vstack(prepared_layers).T
    X = StandardScaler().fit_transform(X)
    for model in models:
        precision = cross_val_score(model, X, y, cv=10, n_jobs=3, scoring='precision')
        recall = cross_val_score(model, X, y, cv=10, n_jobs=3, scoring='recall')
        logging.info("C/V precision for {} is: m = {}, s = {}".format(model.__class__, np.mean(precision), np.std(precision)))
        logging.info("C/V recall for {} is: m = {}, s = {}".format(model.__class__, np.mean(recall), np.std(recall)))
    logging.info("-" * 60)


logging.info("=" * 80)



