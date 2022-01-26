import enum
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import glob
from natsort import natsorted, natsort_keygen
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
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm

print('Loading in metadata')
imgs_location = '../Results/Test_1/data'
metadata = pd.read_csv('../Data/experimentDesign.csv')

sorted_metadata = metadata.copy(deep=True)
sorted_metadata = sorted_metadata.sort_values(by = 'sample_id',key = natsort_keygen())

extracted_metadata = sorted_metadata['tissue'].values

print('Encoding and categorizing labels')
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(extracted_metadata)
train_y = tf.keras.utils.to_categorical(encoded_labels)

num_classes = len(np.unique(encoded_labels))

imgs_list = natsorted(glob.glob(os.path.join(imgs_location,'*.txt')))

print('Loading in data from txt files')
train_X = []
for count in tqdm(range(len(imgs_list))):

    this_img = imgs_list[count]
    temp_img = np.loadtxt(this_img, comments="#", delimiter="\t", unpack=False)

    train_X.append(temp_img)

train_X = np.asarray(train_X)

print('Building model')
model = Sequential([
    tf.keras.layers.Flatten(input_shape = (100,100)),
    tf.keras.layers.Dense(1000,activation='relu'),
    tf.keras.layers.Dense(1000,activation='relu'),
    tf.keras.layers.Dense(num_classes,activation='sigmoid')
])

model = tf.keras.applications.resnet_v2.ResNet50V2(
    include_top=True, weights=None, input_tensor=None,
    input_shape=(100,100,1), pooling=None, classes= num_classes,
    classifier_activation='softmax'
)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy'])

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=250 ,restore_best_weights=True)

# X_train, X_test, y_train, y_test = train_test_split(np.expand_dims(train_X_tabular,axis = 1), categorical_classes, test_size=0.33, random_state=42)

epochs = 250
history = model.fit(
    train_X,train_y,
    validation_split=0.1,
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