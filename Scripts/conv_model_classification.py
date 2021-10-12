import enum
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import glob
from natsort import natsorted
import matplotlib.pyplot as plt
import PIL
import cv2

from tensorflow import keras
from tensorflow.keras import layers, datasets, models
from tensorflow.keras.models import Sequential
import pathlib
import time

from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import LeaveOneOut,KFold, train_test_split

imgs_location = '../Results/Test_2/Data'
data = pd.read_csv('../Data/winequality-red.csv',sep=';')
train_y = data.pop('quality').values

categorical_classes = tf.keras.utils.to_categorical(train_y)

num_classes = len(categorical_classes[0])

imgs_list = natsorted(glob.glob(os.path.join(imgs_location,'*.txt')))

# img0 = np.asarray(PIL.Image.open(str(imgs_list[0])).convert('L'))
img0 = np.loadtxt(imgs_list[0], comments="#", delimiter="\t", unpack=False)

img_height, img_width = img0.shape
img_height = img_width = max([img_height, img_width])
batch_size = 32

def double_expand_dims(np_array):

    expanded_array = np.expand_dims(np.expand_dims(np_array,axis= -1),axis = 0)

    return expanded_array 

def make_dataset(X_data,y_data,n_splits):

    def gen():
        for train_index, test_index in KFold(n_splits).split(X_data):
            X_train, X_test = X_data[train_index], X_data[test_index]
            y_train, y_test = y_data[train_index], y_data[test_index]
            yield X_train,y_train,X_test,y_test

    return tf.data.Dataset.from_generator(gen, (tf.float64,tf.float64,tf.float64,tf.float64))

for count, this_img in enumerate(imgs_list):

    temp_img = np.loadtxt(this_img, comments="#", delimiter="\t", unpack=False)
    temp_img = cv2.resize(temp_img, (img_height,img_width))
    temp_img = double_expand_dims(temp_img)

    if count > 0:
        train_X = np.concatenate([train_X,temp_img],axis = 0)
    else:
        train_X = temp_img

model = Sequential([
    layers.experimental.preprocessing.Rescaling(1./255),
    layers.Conv2D(4, (3,3), padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.2),
    # layers.Conv2D(8, (2), padding='same', activation='relu'),
    # layers.Dropout(0.2),
    # layers.MaxPooling2D(pool_size=(2, 2)),
    # layers.Conv2D(16, (2), padding='same', activation='relu'),
    # layers.MaxPooling2D(),
    # layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes,activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['categorical_accuracy'])

es = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=250 ,restore_best_weights=True)

X_train, X_test, y_train, y_test = train_test_split(train_X, categorical_classes, test_size=0.33, random_state=42)

epochs = 10000
history = model.fit(
    X_train,y_train,
    validation_data = (X_test,y_test),
    epochs=epochs,
    callbacks= [es]
)

for this_key in list(history.history.keys()):
    b = history.history[this_key]
    plt.plot(b,label = this_key)

plt.legend(loc="upper left")
plt.ylim([0,2])
plt.savefig(fname='training_history.png', bbox_inches='tight',pad_inches=0)
plt.show()


print('eof')