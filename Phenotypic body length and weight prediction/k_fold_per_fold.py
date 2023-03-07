import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats as stats
import pickle
import shap
import copy
import matplotlib.pyplot as plt

class datatransform:
    def __init__(self,data):
        self.data = data
        self.X = None
        self.Y = None
        self.X_normalizer = StandardScaler()
        self.Y_normalizer = StandardScaler()

    def transform(self):
        np.random.seed(1024)
        np.random.shuffle(self.data)

        self.X = self.data[:, 1:-1]
        self.Y = self.data[:, -1].reshape(-1, 1)
        self.X = self.X_normalizer.fit_transform(self.X)
        self.Y = self.Y_normalizer.fit_transform(self.Y)
        # return self

def load_data_and_model_Prediction_weight():

    global X_normalizer, Y_normalizer
    df = pd.read_csv('./all_predata.csv', encoding='utf-8', index_col=0)
    with open('./predict_shrimp_body_weight.model', 'rb') as f:
        model = pickle.load(f)
    data = np.hstack((df.loc[:, ["Num"]].values.reshape(-1, 1),
                      df.loc[:,["CL1", "CBH", "CH", "AL3", "AW1", "AW2", "AW3", "AW4",
                                 "AW5", "AW6", "AH1", "AH2", "AH3", "AH4", "AH5", "AH6",
                                 "BOT", "BOS", "CA", "CF", "TS", "EET", "EES", "FO",
                                 "WH", "TT"]].values,
                      # You can select the traits you want to use in the csv file
                      df.loc[:, ["weight"]].values))

    dt = datatransform(data)
    dt.transform()
    return dt,model

def load_data_and_model_Prediction_length():

    global X_normalizer, Y_normalizer
    df = pd.read_csv('./all_predata.csv', encoding='utf-8', index_col=0)
    with open('./predict_shrimp_body_length.model', 'rb') as f:
        model = pickle.load(f)

    data = np.hstack((df.loc[:,["CL1+AL3"]].values.reshape(-1,1),
                      df.loc[:,["BOT","BOS","CA","CF","TS","EET","EES","FO","WH","TT",]].values,
                      # You can select the traits you want to use in the csv file
                      df.loc[:,["BL"]].values))

    dt = datatransform(data)
    dt.transform()
    return dt,model

def k_fold_per_fold(data,model):

    X = data.X
    Y = data.Y
    Y_normalizer=data.Y_normalizer

    kfolder = KFold(n_splits=4, shuffle=True)
    n_samples = np.int64(X.shape[0] - np.floor(X.shape[0] / kfolder.n_splits))

    for test_idx,evl_idx in kfolder.split(X[n_samples:,:]):
        x_evl,  y_evl  = X[n_samples+evl_idx],   Y[n_samples+evl_idx]
        pre = model.predict(x_evl)

        y_pre = Y_normalizer.inverse_transform(pre.reshape(-1,1))
        y_evl = Y_normalizer.inverse_transform(y_evl)
        y_pre = y_pre.ravel(); y_evl = y_evl.ravel()

        print("K-folds:", "mse: %.2f" % (mean_squared_error(y_evl, y_pre)),
              "r: %.2f" % (np.array(stats.pearsonr(y_evl.ravel(), y_pre.ravel())).round(2)[0]),
              "mae: %.2f" % mean_absolute_error(y_evl, y_pre))

if __name__ == '__main__':
    # data, model = load_data_and_model_Prediction_length()
    data, model = load_data_and_model_Prediction_weight()
    k_fold_per_fold(data, model)

