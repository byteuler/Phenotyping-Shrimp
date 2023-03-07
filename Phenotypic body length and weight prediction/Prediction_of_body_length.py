import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error,mean_squared_error
from scipy import stats as stats
import pickle
import copy
import shap

n_samples =1
X_normalizer = StandardScaler()
Y_normalizer = StandardScaler()

def load_data_and_model():
    df = pd.read_csv('./all_predata.csv', encoding='utf-8', index_col=0)
    model = MLPRegressor(hidden_layer_sizes=(10, 10, 10), solver='sgd', alpha=1e-4, max_iter=5000)
    # model = SVR(kernel='rbf', degree=3, gamma='auto', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
    global X_normalizer, Y_normalizer
    data = np.hstack((df.loc[:, ["CL1+AL3"]].values.reshape(-1, 1),
                      df.loc[:, ["BOT", "BOS", "CA", "CF", "TS", "EET", "EES", "FO", "WH", "TT", ]].values,
                      # You can select the traits you want to use in the csv file
                      df.loc[:, ["BL"]].values))
    np.random.seed(1024)
    np.random.shuffle(data)
    X = data[:, 1:-1]
    Y = data[:, -1].reshape(-1, 1)
    X = X_normalizer.fit_transform(X)
    Y = Y_normalizer.fit_transform(Y)

    return X,Y,model

def train_Prediction_of_body_length(X,Y,model):
    Epoch = 4
    last_L1 = np.inf
    kfolder = KFold(n_splits=4, shuffle=True)
    global n_samples, X_normalizer, Y_normalizer
    n_samples = np.int64( X.shape[0] - np.floor(X.shape[0] / kfolder.n_splits))
    for epoch in range(Epoch):
        mae = 0; r = 0;mse = 0;ssr = 0;
        for train_idx, val_idx in kfolder.split(X[:n_samples, :]):

            x_train, y_train = X[train_idx], Y[train_idx]
            x_val, y_val = X[val_idx], Y[val_idx];

            model.fit(x_train, y_train.ravel())
            pre = model.predict(x_val)

            y_pre = Y_normalizer.inverse_transform(pre.reshape(-1, 1))
            y_val = Y_normalizer.inverse_transform(y_val)
            y_pre = y_pre.ravel();
            y_val = y_val.ravel()

            mae += mean_absolute_error(y_val, y_pre)
            mse += mean_squared_error(y_val, y_pre)
            r += np.array(stats.pearsonr(y_val.ravel(), y_pre.ravel())).round(2)

        print('epoch:', epoch,  "r: %.2f" % (r[0] / kfolder.n_splits), "mse: %.2f" % (mse / kfolder.n_splits),
              "mae: %.2f" % (mae / kfolder.n_splits))

        if mae < last_L1:
            last_L1 = mae
            with open('./predict_shrimp_body_length.model', 'wb') as f:
                pickle.dump(model, f)

def test_Prediction_of_body_length(X,Y):

    with open('./predict_shrimp_body_length.model', 'rb') as f:
        load_model = pickle.load(f)
    global n_samples,X_normalizer, Y_normalizer
    x_test = X[n_samples:, :]
    y_test = Y[n_samples:, :]

    pre_test = load_model.predict(x_test)
    pre_test_inv = Y_normalizer.inverse_transform(pre_test)
    y_test = Y_normalizer.inverse_transform(y_test)
    print("Test r: %.2f" % (np.array(stats.pearsonr(y_test.ravel(),pre_test_inv.ravel())[0])),"mse:",mean_squared_error(y_test.ravel(),pre_test_inv).round(2),"mae: %.2f" % mean_absolute_error(y_test, pre_test_inv))

def shape_analysis(X):
    global n_samples
    with open('./predict_shrimp_body_length.model', 'rb') as f:
        load_model = pickle.load(f)

    # X_train_summary = shap.kmeans(X[:n_samples,:], 8)
    explainer = shap.KernelExplainer(load_model.predict, data=X[:n_samples,:])
    x_test = X[n_samples:, :]
    shap_values = explainer.shap_values(x_test)
    shap.summary_plot(copy.deepcopy(shap_values), x_test, feature_names=["BOT","BOS","CA","CF","TS","EET","EES","FO","WH","TT",],show =True)

if __name__ == '__main__':

    X,Y,model = load_data_and_model()
    train_Prediction_of_body_length(X,Y,model)
    test_Prediction_of_body_length(X,Y)
    # shape_analysis(X)


