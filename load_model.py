from AlgoAPI import AlgoAPIUtil, AlgoAPI_Backtest
from datetime import datetime, timedelta
from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from joblib import dump, load
import numpy as np


class AlgoEvent:
    def __init__(self):
        self.one_day_model = None
        self.two_day_model = None
        self.three_day_model = None
        self.arr_X = None
        self.arr_Y = None
        self.today = 0
        self.one_day = 0
        self.two_day = 0
        self.three_day = 0
        self.stated = False
        self.risk_ratio = 0.01
        self.lasttradetime = datetime(2000, 1, 1)

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
        self.two_day_model = load(self.evt.path_lib+'rf_small_two_day.joblib')
        self.three_day_model = load(
            self.evt.path_lib+'rf_small_three_day.joblib')
        self.evt.start()

    def on_bulkdatafeed(self, isSync, bd, ab):
        if isSync:  # and not self.isSaved
            # Get new day price
            if bd[self.Yname[0]]['timestamp'] >= self.lasttradetime + timedelta(hours=24):
                self.lasttradetime = bd[self.myinstrument]['timestamp']

                if not self.stated:
                    self.lasttime = bd[self.Yname[0]]['timestamp']

                    # Append the observation
                    data = np.zeros(len(self.Xname))
                    for i in range(len(self.Xname)):
                        data[i] = bd[self.Xname[i]]['lastPrice']
                    self.arr_Y = (bd[self.Yname[0]]['lastPrice'])
                    self.arr_X = (data)

                    # Predict lastprice for the next three days
                    self.one_day = self.one_day_model.predict(self.arr_X)
                    self.two_day = self.two_day_model.predict(self.arr_X)
                    self.three_day = self.three_day_model.predict(self.arr_X)

                else:
                    # Update date and value
                    self.today = self.one_day
                    self.one_day = self.one_day_model.predict(self.arr_X)
                    self.two_day = self.two_day_model.predict(self.arr_X)
                    self.three_day = self.three_day_model.predict(self.arr_X)

                lastprice = self.arr_Y

                if not np.isnan(self.today) and not np.isnan(self.arr_Y):
                    # Decide on whether to trade # Currently use one day data
                    ideal_position_size = 1 / \
                        np.abs(self.today - self.arr_Y + 1e-6) * \
                        self.risk_ratio
                    ideal_position_size = np.max(ideal_position_size, 0.9)
                    if not np.isnan(ideal_position_size) and ideal_position_size > 0:
                        if self.arr_Y > self.one_day:
                            # buy in ideal position size or buy all
                            self.test_sendOrder(
                                ideal_position_size, lastprice, 1, 'open')
                            pass
                        else:
                            # sell ideal position size or sell all
                            self.test_sendOrder(
                                ideal_position_size, lastprice, -1, 'open')
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

    def test_sendOrder(self, volume, lastprice, buysell, openclose):
        orderObj = AlgoAPIUtil.OrderObject()
        orderObj.instrument = self.Yname
        orderObj.orderRef = 1
        if buysell == 1:
            orderObj.takeProfitLevel = lastprice*1.1
            orderObj.stopLossLevel = lastprice*0.9
        elif buysell == -1:
            orderObj.takeProfitLevel = lastprice*0.9
            orderObj.stopLossLevel = lastprice*1.1
        orderObj.volume = volume
        orderObj.openclose = openclose
        orderObj.buysell = buysell
        orderObj.ordertype = 0  # 0=market_order, 1=limit_order
        self.evt.sendOrder(orderObj)
