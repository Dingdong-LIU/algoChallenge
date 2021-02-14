from AlgoAPI import AlgoAPIUtil, AlgoAPI_Backtest
from datetime import datetime, timedelta
from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def get_dataset(X_train, y_train, names):
    arr_X = X_train.copy()
    arr_Y = y_train.copy()
    arr_X = arr_X[2:]
    arr_Y = arr_Y[:-2]
    arr_X = pd.DataFrame(arr_X, columns=names)
    return arr_X, arr_Y


class AlgoEvent:
    def __init__(self):
        self.lasttime = datetime(2000, 1, 1)
        self.isSaved = False
        self.firstTrain = True
        self.numOfObs = 400
        self.history = None
        self.dict = {}
        self.arr_Y, self.arr_X = [], []

        # get my selected financial instruments
        self.Xname = ["ETXEUR", "FRXEUR", "GRXEUR", "HKXHKD", "NLXEUR", "NSXUSD",
                      "SPXUSD", "UKXGBP", "US2000USD", "US30USD", "BCOUSD", "CORNUSD",
                      "NATGASUSD", "SOYBNUSD", "SUGARUSD", "WHEATUSD", "WTIUSD", "XAGAUD",
                      "XAGUSD", "XAGEUR", "XAUEUR", "XAUUSD", "XCUUSD", "XPTUSD", "XPDUSD",
                      "XAGHKD", "XAGJPY", "XAUXAG"]
        self.Yname = ["EURUSD"]
        self.params = {'n_estimators': 1200,
                       'max_depth': 3,
                       'min_samples_split': 5,
                       'learning_rate': 0.01,
                       'loss': 'ls'}
        self.input_dim = len(self.Xname)

        pass

    def start(self, mEvt):

        self.evt = AlgoAPI_Backtest.AlgoEvtHandler(self, mEvt)
        self.evt.start()

    def on_bulkdatafeed(self, isSync, bd, ab):
        if isSync:  # and not self.isSaved
            # Get new day price
            if bd[self.Yname[0]]['timestamp'] > self.lasttime + timedelta(hours=24):
                self.lasttime = bd[self.Yname[0]]['timestamp']

                # Append the observation
                data = np.zeros(len(self.Xname))
                for i in range(len(self.Xname)):
                    data[i] = bd[self.Xname[i]]['lastPrice']
                self.arr_Y.append(bd[self.Yname[0]]['lastPrice'])
                self.arr_X.append(data)

                if len(self.arr_Y) >= self.numOfObs:
                    # Do the training

                    # get some parameters and dataset
                    params = self.params
                    names = self.Xname

                    arr_X, arr_Y = get_dataset(
                        self.arr_X, self.arr_Y, self.Xname)

                    # split dataset for train and validation
                    X_train, X_test, y_train, y_test = train_test_split(
                        arr_X, arr_Y, test_size=len(arr_X)//5, random_state=13)

                    reg = ensemble.GradientBoostingRegressor(**(self.params))

                    reg.fit(X_train, y_train)

                    mse = mean_squared_error(y_test, reg.predict(X_test))
                    self.evt.consoleLog(
                        "The mean squared error (MSE) on test set: {:.6f}".format(mse))
                    r2 = r2_score(y_test, reg.predict(X_test))
                    self.evt.consoleLog(
                        "The R2 score on the test set is: {:.6f}".format(r2))

                    # plot the image
                    test_score = np.zeros(
                        (params['n_estimators'],), dtype=np.float64)

                    for i, y_pred in enumerate(reg.staged_predict(X_test)):
                        test_score[i] = reg.loss_(y_test, y_pred)

                    fig = plt.figure(figsize=(6, 6))
                    plt.subplot(1, 1, 1)
                    plt.title('Deviance')
                    plt.plot(np.arange(params['n_estimators']) + 1, reg.train_score_, 'b-',
                             label='Training Set Deviance')
                    plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
                             label='Test Set Deviance')
                    plt.legend(loc='upper right')
                    plt.xlabel('Boosting Iterations')
                    plt.ylabel('Deviance')
                    fig.tight_layout()
                    plt.savefig(self.evt.path_img + "a.png")

                    # plot another image

                    feature_importance = reg.feature_importances_

                    sorted_idx = np.argsort(feature_importance)
                    pos = np.arange(sorted_idx.shape[0]) + .5
                    fig = plt.figure(figsize=(12, 6))
                    plt.subplot(1, 2, 1)
                    plt.barh(
                        pos, feature_importance[sorted_idx], align='center')
                    plt.yticks(pos, np.array(names)[sorted_idx])
                    plt.title('Feature Importance (MDI)')

                    result = permutation_importance(reg, X_test, y_test, n_repeats=10,
                                                    random_state=42, n_jobs=2)
                    sorted_idx = result.importances_mean.argsort()
                    plt.subplot(1, 2, 2)
                    plt.boxplot(result.importances[sorted_idx].T,
                                vert=False, labels=np.array(names)[sorted_idx])
                    plt.title("Permutation Importance (test set)")
                    fig.tight_layout()
                    plt.savefig(self.evt.path_img + "b.png")

                    # Report number of instances
                    self.evt.consoleLog(
                        "The number of instances used: {}".format(self.numOfObs))
                    self.numOfObs += 50

                    # Save the model
                    self.isSaved = True

        pass

    def on_marketdatafeed(self, md, ab):
        pass

    def on_newsdatafeed(self, nd):
        pass

    def on_weatherdatafeed(self, wd):
        pass

    def on_econsdatafeed(self, ed):
        pass

    def on_orderfeed(self, of):
        pass

    def on_dailyPLfeed(self, pl):
        pass

    def on_openPositionfeed(self, op, oo, uo):
        pass
