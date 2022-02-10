param_hotel = "City Hotel"
param_month = "2"
param_num = "12"

param_num = int(param_num)

import pickle
import numpy as np

with open('exported_one_hot.pickle', 'rb') as fp:
    enc = pickle.load(fp)

with open('exported_classifier.pickle', 'rb') as fp:
    classifier = pickle.load(fp)

hotel_feature = enc.transform([[param_hotel]]).toarray()
month_featrue = (int(param_month) >= 6) and (int(param_month) <= 8)

features = np.hstack([
    hotel_feature,
    np.array([[month_featrue]]),
    np.array([[param_num]])]
)

if classifier.predict(features)[0]:
    print("will not cancel")
else:
    print("will cancel")