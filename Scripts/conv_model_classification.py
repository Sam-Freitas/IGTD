import enum
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import glob
from natsort import natsorted
import matplotlib.pyplot as plt
import PIL

from tensorflow import keras
from tensorflow.keras import layers, datasets, models
from tensorflow.keras.models import Sequential
import pathlib
import time

from tensorflow.python.keras.callbacks import EarlyStopping

imgs_location = '../Results/Test_1/Data'
data = pd.read_csv('../Data/winequality-red.csv',sep=';')
train_y = data.pop('quality').values

categorical_classes = tf.keras.utils.to_categorical(train_y)

num_classes = len(categorical_classes[0])

imgs_list = natsorted(glob.glob(os.path.join(imgs_location,'*.txt')))

# img0 = np.asarray(PIL.Image.open(str(imgs_list[0])).convert('L'))
img0 = np.loadtxt(imgs_list[0], comments="#", delimiter="\t", unpack=False)

img_height, img_width = img0.shape
batch_size = 32

def double_expand_dims(np_array):

    expanded_array = np.expand_dims(np.expand_dims(np_array,axis= -1),axis = 0)

    return expanded_array 

for count, this_img in enumerate(imgs_list):

    temp_img = np.loadtxt(this_img, comments="#", delimiter="\t", unpack=False)
    temp_img = double_expand_dims(temp_img/255)

    if count > 0:
        train_X = np.concatenate([train_X,temp_img],axis = 0)
    else:
        train_X = temp_img

# setup the model
model = Sequential([
    layers.experimental.preprocessing.Rescaling(1./255),
    layers.Conv2D(16, (3,3), padding='same', activation='relu'),
    # layers.MaxPooling2D(),
    # layers.Conv2D(8, (2), padding='same', activation='relu'),
    # layers.MaxPooling2D(),
    # layers.Conv2D(16, (2), padding='same', activation='relu'),
    # layers.MaxPooling2D(),
    # layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes,activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['BinaryCrossentropy','CategoricalAccuracy'])

es = tf.keras.callbacks.EarlyStopping(monitor='categorical_accuracy', patience=100 ,mode='max',
    restore_best_weights=True)

epochs = 1000
history = model.fit(
    train_X,categorical_classes,
    epochs=epochs,
    callbacks= [es]
)

print('eof')