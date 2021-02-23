# initial set up
import pickle
import tensorflow as tf
import tensorflow.keras
import numpy as np
import os
from glob import glob
import cv2
from sklearn.model_selection import train_test_split
from PIL import Image
import skimage
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import pandas as pd
import collections
from imblearn.over_sampling import RandomOverSampler

# setting a gpu
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpus= tf.config.list_physical_devices('GPU')
print(gpus)
for i in range(len(gpus)):
	tf.config.experimental.set_memory_growth(gpus[i], True)

# define plot process function
def plot_process(acc, val_acc, loss, val_loss):
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
    plt.show()

def preprocessing_resize(pixels):
    a = []
    
    for i in range(len(pixels)):
            image_string = (pixels)[i].split(' ') 
            image_data = np.asarray(image_string, dtype=np.uint8).reshape(48,48,1)
            image_data = cv2.resize(image_data, dsize=(160, 160), interpolation=cv2.INTER_CUBIC)
            image_data = image_data.reshape(160, 160, 1)
            image_data2 = cv2.merge((image_data, image_data, image_data))
            a.append(image_data2)

    return a

def load_dataset2():
    train_csv = pd.read_csv('../Facial expression/data/data_b/train.csv')
    
    oversample = RandomOverSampler(sampling_strategy='auto')

    X_over, y_over = oversample.fit_resample((train_csv.pixels).values.reshape(-1, 1), train_csv.emotion)

    a = np.array(y_over)
    
    y_over = pd.Series(y_over)
    y_over= y_over.values.reshape(len(y_over),1)
    
    y_over = to_categorical(y_over)

    X_train,X_val,y_train,y_val = train_test_split(X_over,y_over, test_size=0.2)
    
    X_train = pd.Series(X_train.flatten())
    X_train = np.array(preprocessing_resize(X_train))
    
    X_val = pd.Series(X_val.flatten())
    X_val = np.array(preprocessing_resize(X_val))
    
    
    print ("X_train shape: " + str(X_train.shape))
    print ("y_train shape: " + str(y_train.shape))
    print ("X_val shape: " + str(X_val.shape))
    print ("y_val shape: " + str(y_val.shape))
    return X_train, y_train, X_val, y_val
    #return X_train[:5000], y_train[:5000], X_val[:2000], y_val[:2000]


X_train, y_train, X_val, y_val = load_dataset2()
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))

BATCH_SIZE = 256
SHUFFLE_BUFFER_SIZE = 100

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
val_dataset = val_dataset.batch(BATCH_SIZE)

IMG_SHAPE = (160, 160) + (3,)
base_model = tf.keras.applications.MobileNet(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False

test_model = tf.keras.models.Sequential()
num_layers = 74
#test_model.add(tf.keras.layers.InputLayer(input_shape=(160, 160, 3), name = 'haha'))
# for i in range(num_layers):
#     test_model.add(base_model.layers[i])
#     test_model.layers[i].trainable = False
#test_model.add(base_model)
for layer in base_model.layers:
    if layer.name == 'conv_dw_9':
        break
    test_model.add(layer)

train_bool = False
for layer in test_model.layers:
    if layer.name == 'conv_pw_8_relu':
        train_bool = True
    layer.trainable = train_bool
test_model.add(tf.keras.layers.GlobalAveragePooling2D())
test_model.add(tf.keras.layers.BatchNormalization())
# test_model.add(tf.keras.layers.Dense(128, activation='relu'))
test_model.add(tf.keras.layers.Dense(7, activation="softmax"))

# test_model.trainable = True

base_learning_rate = 0.0001
test_model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])
test_model.summary()

loss0, accuracy0 = test_model.evaluate(val_dataset)
print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

initial_epochs = 30
history = test_model.fit(train_dataset,
                    epochs=initial_epochs,
                    validation_data=val_dataset)

histories={}
histories[1] = history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']
plot_process(acc, val_acc, loss, val_loss)

test_model.save_weights('weight1.h5')

for layer in test_model.layers:
    layer.trainable = True


test_model.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy'])
test_model.summary()

one_more = 90
total_epochs = initial_epochs + one_more
history = test_model.fit(train_dataset,
                         epochs=total_epochs,
                         initial_epoch=history.epoch[-1],
                         validation_data=val_dataset)

histories[2] = history
acc += history.history['accuracy']
val_acc += history.history['val_accuracy']

loss += history.history['loss']
val_loss += history.history['val_loss']

dict = {"acc": acc, "val_acc": val_acc, "loss": loss, "val_loss": val_loss}
f = open("../training_2/accurarcy-12-05-05.pkl", "wb")
pickle.dump(dict, f)
f.close()

test_model.save_weights('weight1-2.h5')
