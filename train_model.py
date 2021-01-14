
from AlgoAPI import AlgoAPIUtil, AlgoAPI_Backtest
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np


def build_model(input_dim):
    batch_size = 128
    units = 256
    output_size = 1
    lstm_layer = tf.keras.layers.LSTM(units, input_shape=(None, input_dim))
    model = keras.models.Sequential(
        [
            lstm_layer,
            keras.layers.BatchNormalization(),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(output_size),
        ]
    )
    # define a model
    return model


def get_dataset(X_train, y_train):
    arr_X = []
    arr_Y = y_train.copy()
    time = 3
    for i in range(len(X_train)):
        if i > 0:
            if y_train[i] > y_train[i-1]:
                arr_Y[i] = 1
            else:
                arr_Y[i] = 0
        x = np.zeros((time, (X_train[0].shape)[0]))
        if i >= time-1:
            for j in range(time):
                x[j] = X_train[j+i-6]
            arr_X.append(x)
    arr_X.pop(0)
    arr_Y = arr_Y[time:]
    return arr_X, arr_Y


class AlgoEvent:
    def __init__(self):
        self.lasttime = datetime(2000, 1, 1)
        self.isSaved = False
        self.firstTrain = True
        self.numOfObs = 100
        self.history = None
        self.dict = {}
        self.arr_Y, self.arr_X = [], []

        # get my selected financial instruments
        self.Xname = ["ETXEUR", "FRXEUR", "GRXEUR", "HKXHKD", "NLXEUR", "NSXUSD", "SPXUSD",
                      "UKXGBP", "US2000USD", "US30USD"]
        self.Yname = ["EURUSD"]

        input_dim = len(self.Xname)

        self.model = build_model(input_dim)
        self.model.compile(
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
            optimizer="adam",
            metrics=["accuracy"],
        )

        pass

    def start(self, mEvt):

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
                    arr_X, arr_Y = get_dataset(self.arr_X, self.arr_Y)

                    cut_point = len(arr_X)//5*4
                    arr_X_train = np.array(arr_X[:cut_point])
                    arr_Y_train = np.array(arr_Y[:cut_point])
                    arr_X_val = np.array(arr_X[cut_point:])
                    arr_Y_val = np.array(arr_Y[cut_point:])

                    if self.firstTrain:
                        self.firstTrain = False
                        self.history = self.model.fit(arr_X_train, arr_Y_train,
                                                      validation_data=(
                                                          arr_X_val, arr_Y_val), batch_size=128,
                                                      epochs=30)

                        self.dict = {"loss": self.history.history['loss'], "val_loss": self.history.history['val_loss'],
                                     "accuracy": self.history.history['accuracy'], "val_accuracy": self.history.history['val_accuracy']}

                    else:
                        self.history = self.model.fit(arr_X_train, arr_Y_train,
                                                      validation_data=(
                                                          arr_X_val, arr_Y_val), batch_size=128,
                                                      epochs=30, initial_epoch=self.history.epoch[-1])
                        self.dict['loss'] += self.history.history['loss']
                        self.dict['val_loss'] += self.history.history['val_loss']
                        self.dict['accuracy'] += self.history.history['accuracy']
                        self.dict['val_accuracy'] += self.history.history['val_accuracy']

                    acc = self.dict['accuracy']
                    val_acc = self.dict['val_accuracy']
                    loss = self.dict['loss']
                    val_loss = self.dict['val_loss']

                    plt.figure(figsize=(8, 8))
                    plt.subplot(2, 1, 1)
                    plt.plot(acc, label='Training Accuracy')
                    plt.plot(val_acc, label='Validation Accuracy')
                    plt.legend(loc='lower right')
                    plt.ylabel('Accuracy')

                    plt.title('Training and Validation Accuracy')

                    plt.subplot(2, 1, 2)
                    plt.plot(loss, label='Training Loss')
                    plt.plot(val_loss, label='Validation Loss')
                    plt.legend(loc='upper right')
                    plt.ylabel('Cross Entropy')

                    plt.title('Training and Validation Loss')
                    plt.xlabel('epoch')

                    plt.savefig(self.evt.path_img + "a.png")

                    # Save the model
                    self.model.save(self.evt.path_lib+"lstm_model_1")
                    self.evt.consoleLog("successfully saved model")
                    self.evt.consoleLog(
                        "val_acc, train_acc = ", val_acc[-1], acc[-1])
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
