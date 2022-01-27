import enum
from msilib.schema import _Validation_records
from tabnanny import verbose
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import glob
from natsort import natsorted, natsort_keygen
import matplotlib.pyplot as plt
import PIL
import cv2
import json

from tensorflow import keras
from tensorflow.keras import layers, datasets, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import pathlib
import time

from sklearn.model_selection import LeaveOneOut,KFold, train_test_split,StratifiedKFold
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
    temp_img = np.loadtxt(this_img, comments="#", delimiter="\t", unpack=False).astype(np.uint8)

    train_X.append(temp_img)

train_X = np.asarray(train_X)

skf = StratifiedKFold(n_splits=5)

print('Building model')
# model = Sequential([
#     tf.keras.layers.Flatten(input_shape = (100,100)),
#     tf.keras.layers.Dense(1000,activation='relu'),
#     tf.keras.layers.Dense(1000,activation='relu'),
#     tf.keras.layers.Dense(num_classes,activation='sigmoid')
# ])

kfold_counter = 1
for train_index, test_index in skf.split(train_X, encoded_labels):

    model = tf.keras.applications.resnet_v2.ResNet50V2(
    include_top=True, weights=None, input_tensor=None,
    input_shape=(100,100,1), pooling=None, classes= num_classes,
    classifier_activation='softmax'
    )

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy'])

    checkpoint_path = "training_weights_kfold_" + str(kfold_counter) + "/cp.ckpt" 
    checkpoint_dir = os.path.dirname(checkpoint_path)

    checkpoint = ModelCheckpoint(filepath = checkpoint_path,monitor = "val_accuracy", mode = 'max',
        save_best_only = True,verbose=1,save_weights_only=True) #use checkpoint instead of sequential() module
    earlystop = EarlyStopping(monitor = 'val_loss', min_delta=0.00001,
        patience = 25, verbose = 1,restore_best_weights = True) #stop at best epoch
    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor=0.5,
        patience=8, min_lr=0.0001, verbose = 1)

    kfold_X = train_X[train_index]
    kfold_y = train_y[train_index]

    kfold_test = (train_X[test_index],train_y[test_index])

    epochs = 150
    history = model.fit(
        kfold_X,kfold_y,
        validation_data = kfold_test,
        epochs=epochs,
        batch_size = 16,
        callbacks= [earlystop, checkpoint,reduce_lr]
    )

    eval_result = model.evaluate(train_X,train_y,steps=1,return_dict = True,batch_size = 4) #get evaluation results
    res = dict()
    for key in eval_result: res[key] = round(eval_result[key], 5)

    print(res)

    for this_key in list(history.history.keys()):
        b = history.history[this_key]
        plt.plot(b,label = this_key)

    plt.legend(loc="upper left")
    plt.ylim([0,2])
    plt.title(json.dumps(res))
    plt.savefig(fname="training_history_" + str(kfold_counter) + ".png", bbox_inches='tight',pad_inches=0)
    plt.show()

    plt.close('all')
    kfold_counter = kfold_counter + 1

    del model

print('eof')