import enum
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import glob
from natsort import natsorted
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
import time

imgs_location = '../Results/Test_1/Data'
data = pd.read_csv('../Data/winequality-red.csv',sep=';')
train_y = data.pop('quality').values

categorical_classes = tf.keras.utils.to_categorical(train_y)

num_classes = len(np.unique(train_y))

imgs_list = natsorted(glob.glob(os.path.join(imgs_location,'*.txt')))

img0 = np.loadtxt(imgs_list[0], comments="#", delimiter="\t", unpack=False)

img_height, img_width = img0.shape
batch_size = 32


train_X = []
for count, this_img in enumerate(imgs_list):

    temp_img = np.loadtxt(this_img, comments="#", delimiter="\t", unpack=False)
    temp_img = temp_img/255

    train_X.append(train_X)

    if count > 100:
        break

# setup the model
model = Sequential([
    layers.experimental.preprocessing.Rescaling(1./255),
    layers.Conv2D(16, (3,3), padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, (3,3), padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='tanh'),
    layers.Dense(num_classes)
])

model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

epochs = 5
history = model.fit(
  train_X,train_y,
  epochs=epochs,
)

print('eof')