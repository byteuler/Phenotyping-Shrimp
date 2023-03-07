import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from  sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import KFold,GroupKFold,StratifiedKFold
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,RobustScaler
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats as stats
# from sklearn.externals import joblib
import pickle
import shap
import copy
import matplotlib.pyplot as plt




n_samples =1
X_normalizer = StandardScaler()
Y_normalizer = StandardScaler()

def load_data_and_model():
    df = pd.read_csv('./all_predata.csv', encoding='utf-8', index_col=0)
    model = MLPRegressor(hidden_layer_sizes=(20, 20, 20), solver='sgd', alpha=1e-4, max_iter=5000)
    # model  = SVR(kernel ='rbf',degree = 3,gamma ='auto_deprecated',coef0 = 0.0,tol = 0.001,C = 1.0,epsilon = 0.1,shrinking = True,cache_size = 200,verbose = False,max_iter = -1 )
    data = np.hstack((df.loc[:, ["Num"]].values.reshape(-1, 1),
                       df.loc[:,["CL1", "CBH", "CH", "AL3", "AW1", "AW2", "AW3", "AW4",\
                                "AW5", "AW6", "AH1", "AH2", "AH3", "AH4", "AH5",\
                                "AH6", "BOT", "BOS", "CA", "CF", "TS", "EET", "EES",\
                                "FO", "WH", "TT"]].values,
                       df.loc[:, ["weight"]].values))
    # You can select the traits you want to use in the csv file
    np.random.seed(1024)
    np.random.shuffle(data)
    X = data[:, 1:-1]
    Y = data[:, -1].reshape(-1, 1)
    global X_normalizer, Y_normalizer
    X = X_normalizer.fit_transform(X)
    Y = Y_normalizer.fit_transform(Y)

    return X,Y,model

def train_Prediction_of_body_weight(X, Y, model):
    global n_samples, X_normalizer, Y_normalizer
    Epoch = 4
    last_L1 = np.inf
    kfolder = KFold(n_splits=4, shuffle=True)
    n_samples = np.int64(X.shape[0] - np.floor(X.shape[0] / kfolder.n_splits))
    for epoch in range(Epoch):
        mae = 0; r = 0;mse = 0;
        for train_idx,val_idx in kfolder.split(X[:n_samples,:]):
            x_train, y_train = X[train_idx], Y[train_idx]
            x_val,  y_val  = X[val_idx],   Y[val_idx]
            model.fit(x_train, y_train.ravel())
            pre = model.predict(x_val)

            y_pre = Y_normalizer.inverse_transform(pre.reshape(-1,1))
            y_val = Y_normalizer.inverse_transform(y_val)

            y_pre = y_pre.ravel(); y_val = y_val.ravel()

            mae += mean_absolute_error(y_val, y_pre)
            mse += mean_squared_error(y_val,y_pre)
            r += np.array(stats.pearsonr(y_val,y_pre)).round(2)

        print('epoch:', epoch, "r: %.2f" % (r[0] / kfolder.n_splits), "mse: %.2f" % (mse / kfolder.n_splits),
              "mae: %.2f" % (mae / kfolder.n_splits))

        if mae < last_L1:
           last_L1 = mae
           with open('./predict_shrimp_body_weight.model','wb') as f:
                pickle.dump(model,f)
def test_Prediction_of_body_weight(X,Y):
    global n_samples, X_normalizer, Y_normalizer
    with open('./predict_shrimp_body_weight.model','rb') as f:
        load_model = pickle.load(f)


    x_test=X[n_samples:,:]
    y_test=Y[n_samples:,:]

    pre_test = load_model.predict(x_test)
    pre_test_inv = Y_normalizer.inverse_transform(pre_test)
    y_test = Y_normalizer.inverse_transform(y_test)
    print("Test r: %.2f" % (np.array(stats.pearsonr(y_test.ravel(),pre_test_inv.ravel())[0])),"mse:",mean_squared_error(y_test.ravel(),pre_test_inv).round(2),"mae: %.2f" % mean_absolute_error(y_test, pre_test_inv))


def shape_analysis(X):
    global n_samples
    with open('./predict_shrimp_body_weight.model', 'rb') as f:
        load_model = pickle.load(f)
    x_test=X[n_samples:,:]

    # X_train_summary = shap.kmeans(X[:n_samples,:], 8)
    explainer = shap.KernelExplainer(load_model.predict, data=X[:n_samples,:])
    shap_values = explainer.shap_values(x_test)
    shap.summary_plot(copy.deepcopy(shap_values), x_test, max_display=26,feature_names=["CL1","CBH","CH","AL3","AW1","AW2","AW3","AW4","AW5","AW6","AH1","AH2","AH3","AH4","AH5","AH6", "BOT","BOS","CA","CF","TS","EET","EES","FO","WH","TT"],show =True)

if __name__ == '__main__':


    X,Y,model = load_data_and_model()
    train_Prediction_of_body_weight(X,Y,model)
    test_Prediction_of_body_weight(X,Y)
    # shape_analysis(X)

