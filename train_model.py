from AlgoAPI import AlgoAPIUtil, AlgoAPI_Backtest
from datetime import datetime, timedelta
import tensorflow as tf
import matplot.plt as plt
import numpy as np


def get_model():
    model = tf.Sequential()
    # define a model
    return model

class AlgoEvent:
    def __init__(self):
        self.lasttime = datetime(2000, 1, 1)
        self.isSaved = False
        self.numOfObs = 100
        self.arr_Y, self.arr_X = [], []
        self.model = get_model()
        pass
        
    def start(self, mEvt):
        # get my selected financial instruments
        self.Xname = ["ETXEUR", "FRXEUR", "GRXEUR", "HKXHKD", "NLXEUR", "NSXUSD", "SPXUSD", "UDXUSD", 
            "UKXGBP", "US2000USD", "US30USD"]
        self.Yname = ["EURUSD"]
        
        
        self.evt = AlgoAPI_Backtest.AlgoEvtHandler(self, mEvt)
        self.evt.start()
        
    def on_bulkdatafeed(self, isSync, bd, ab):
        if isSync and not self.isSaved:
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
                    
                    
                    
                    # Save the model
                    self.model.save(self.evt.path_lib+"lstm_model_1")
                    
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