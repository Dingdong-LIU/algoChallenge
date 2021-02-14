from AlgoAPI import AlgoAPIUtil, AlgoAPI_Backtest
from datetime import datetime, timedelta
from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from joblib import dump, load
import numpy as np
import pandas as pd


def get_dataset(X_train, y_train, names):
    arr_X = X_train.copy()
    arr_Y = y_train.copy()
    arr_X = arr_X[1:]
    arr_Y = arr_Y[:-1]
    arr_X = pd.DataFrame(arr_X, columns=names)
    return arr_X, arr_Y


class AlgoEvent:
    def __init__(self):
        self.one_day_model = None
        self.two_day_model = None
        self.three_day_model = None
        self.arr_X = None
        self.arr_Y = None
        self.today = 0
        self.one_day = -10
        self.two_day = -10
        self.three_day = -10
        self.stated = False
        self.risk_ratio = 0.01
        self.lasttradetime = datetime(2000, 1, 1)
        self.volumn = 0.0
        self.trend = False  # false is keep decreasing, true is keep increasing

        self.X_train = []
        self.y_train = []
        self.numOfObs = 30
        self.reg = None
        self.params = {'n_estimators': 800,
                       'max_depth': 3,
                       'min_samples_split': 5,
                       'learning_rate': 0.01,
                       'loss': 'ls'}

        self.selected_model = self.one_day_model
        pass

    def start(self, mEvt):
        self.evt = AlgoAPI_Backtest.AlgoEvtHandler(self, mEvt)

        self.Xname = ["ETXEUR", "GRXEUR", "HKXHKD", "NSXUSD",
                      "SPXUSD", "US2000USD", "US30USD", "CORNUSD",
                      "NATGASUSD", "SOYBNUSD", "SUGARUSD", "WHEATUSD", "WTIUSD",
                      "XAGEUR", "XAUEUR", "XAUUSD", "XCUUSD", "XPTUSD", "XPDUSD",
                      "XAUXAG"]
        self.Yname = ["EURUSD"]
        self.one_day_model = load(self.evt.path_lib+'rf_small_one_day.joblib')
        # self.two_day_model = load(self.evt.path_lib+'rf_small_two_day.joblib')
        # self.three_day_model = load(
        #     self.evt.path_lib+'rf_small_three_day.joblib')
        self.selected_model = self.one_day_model
        self.evt.start()

    def on_bulkdatafeed(self, isSync, bd, ab):
        if isSync:  # and not self.isSaved
            # Get new day price
            if bd[self.Yname[0]]['timestamp'] >= self.lasttradetime + timedelta(hours=24):
                r2 = 0
                self.lasttradetime = bd[self.Yname[0]]['timestamp']

                # Append the observation
                data = np.zeros(len(self.Xname))

                for i in range(len(self.Xname)):
                    data[i] = bd[self.Xname[i]]['lastPrice']
                self.arr_Y = bd[self.Yname[0]]['lastPrice']
                self.y_train.append(bd[self.Yname[0]]['lastPrice'])
                self.arr_X = pd.DataFrame([data], columns=self.Xname)
                self.X_train.append(data)

                if len(self.y_train) > self.numOfObs:
                    arr_X, arr_Y = get_dataset(
                        self.X_train, self.y_train, self.Xname)
                    X_train, X_test, y_train, y_test = train_test_split(
                        arr_X, arr_Y, test_size=len(arr_X)//5, random_state=13)
                    self.reg = ensemble.GradientBoostingRegressor(
                        **(self.params))
                    self.reg.fit(X_train, y_train)
                    r2 = r2_score(y_test, self.reg.predict(X_test))
                    r2_original = r2_score(
                        y_test, self.one_day_model.predict(X_test))
                    self.numOfObs += 30
                    if r2 > r2_original:
                        self.evt.consoleLog("selcted new model")
                        self.selected_model = self.reg
                    else:
                        self.evt.consoleLog("selected mature model")
                        self.selected_model = self.one_day_model
                    pass

                if not np.isnan(self.arr_Y):
                    if not self.stated:
                        # Predict lastprice for the next three days
                        self.one_day = self.selected_model.predict(self.arr_X)
                        # self.two_day = self.two_day_model.predict(self.arr_X)
                        # self.three_day = self.three_day_model.predict(
                        #     self.arr_X)

                    else:
                        # Update date and value
                        self.today = self.one_day
                        self.one_day = self.selected_model.predict(self.arr_X)
                        # self.two_day = self.two_day_model.predict(self.arr_X)
                        # self.three_day = self.three_day_model.predict(
                        #     self.arr_X)

                    lastprice = self.arr_Y

                # problem: start  with a minimum or a maximum
                if self.stated == False:
                    self.stated = True      # Do not allow trading in the first day
                    pass
                elif not np.isnan(self.today) and not np.isnan(self.arr_Y):
                    # Decide on whether to trade # Currently use one day data
                    ideal_position_size = 1 / \
                        np.abs(self.today - self.arr_Y + 1e-6) * \
                        self.risk_ratio
                    self.volumn = np.maximum(ideal_position_size, 0)
                    lastprice = self.arr_Y
                    if not np.isnan(ideal_position_size) and ideal_position_size > 0:
                        self.evt.consoleLog(
                            "self.one_day = {}, self.arr_Y = {}".format(self.one_day, self.arr_Y))
                        if self.one_day > self.arr_Y and not self.trend:
                            # buy in ideal position size or buy all
                            self.test_sendOrder(lastprice, 1, 'open')
                            pass
                        elif self.one_day < self.arr_Y and self.trend:
                            # sell ideal position size or sell all
                            self.test_sendOrder(lastprice, -1, 'open')
                            pass
                        self.trend = self.one_day > self.arr_Y

                self.arr_X = None
                self.arr_Y = None

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

    def test_sendOrder(self, lastprice, buysell, openclose):
        orderObj = AlgoAPIUtil.OrderObject()
        orderObj.instrument = self.Yname[0]
        orderObj.orderRef = 1
        # if buysell == 1:
        #     orderObj.takeProfitLevel = lastprice*1.1
        #     orderObj.stopLossLevel = lastprice*0.9
        # elif buysell == -1:
        #     orderObj.takeProfitLevel = lastprice*0.9
        #     orderObj.stopLossLevel = lastprice*1.1
        orderObj.volume = float(self.volumn)
        orderObj.openclose = openclose
        orderObj.buysell = buysell
        orderObj.ordertype = 0  # 0=market_order, 1=limit_order
        self.evt.sendOrder(orderObj)


# Notes: set leverage to 10 and only stoploss value with time 2016.01-2018.01 win 0.17% daily
