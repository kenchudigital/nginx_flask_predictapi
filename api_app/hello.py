from flask import Flask
from flask import request

import pickle
import numpy as np

app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def hello_world():

    param_hotel = request.form.get("hotel_type")
    param_month = request.form.get("arrival_month")
    param_num = request.form.get("number_of_people")

    if param_hotel is None:
        return "Please provide hotel type."
    if param_month is None:
        return "Please provide arrival month."
    if param_num is None:
        return "Please provide number of resident."
    if not param_month.isdigit():
        return "Month must be integer."
    if not param_num.isdigit():
        return "Number must be integer."
    if param_hotel == "city":
        param_hotel = "City Hotel"
    elif param_hotel == "resort":
        param_hotel = "Resort Hotel"
    else:
        return "Hotel type must be [city, resort]"

    param_month = int(param_month)
    param_num = int(param_num)

    with open('exported_one_hot.pickle', 'rb') as fp:
        enc = pickle.load(fp)

    with open('exported_classifier.pickle', 'rb') as fp:
        classifier = pickle.load(fp)

    hotel_feature = enc.transform([[param_hotel]]).toarray()
    month_featrue = (param_month >= 6) and (param_month <= 8)

    features = np.hstack([
        hotel_feature,
        np.array([[month_featrue]]),
        np.array([[param_num]])]
    )

    if classifier.predict(features):
        return "will not cancel"
    else:
        return "will cancel"
