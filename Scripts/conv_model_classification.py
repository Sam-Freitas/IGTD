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
from tensorflow.keras import layers, datasets, models, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import pathlib
import time

from sklearn.model_selection import LeaveOneOut,KFold, train_test_split,StratifiedKFold,StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm

print('Loading in metadata')
imgs_location = '../Results/Test_2_100/data'
metadata = pd.read_csv('../Data/experimentDesign.csv')

sorted_metadata = metadata.copy(deep=True)
sorted_metadata = sorted_metadata.sort_values(by = 'sample_id',key = natsort_keygen())

# # attempt with using single tissue data
# single_tissue_index = (sorted_metadata['tissue'].values == 'Skin')
# metadata_single_tissue = sorted_metadata.iloc[single_tissue_index]
# extracted_metadata = metadata_single_tissue['age'].values
# encoded_labels = extracted_metadata
# train_y = extracted_metadata
# num_classes = 1

# # categorical age prediction 
# print('Encoding and categorizing labels')
# label_encoder = LabelEncoder()
# encoded_labels = label_encoder.fit_transform(extracted_metadata)
# train_y = tf.keras.utils.to_categorical(encoded_labels)
# num_classes = len(np.unique(encoded_labels))

# # basic take everything and get age
extracted_metadata = sorted_metadata['age'].values
encoded_labels = extracted_metadata
train_y = extracted_metadata
num_classes = 1

imgs_list = natsorted(glob.glob(os.path.join(imgs_location,'*.txt')))

print('Loading in data from txt files')
train_X = []
for count in tqdm(range(len(imgs_list))):

    this_img = imgs_list[count]
    # # if just loadining in everything
    temp_img = np.loadtxt(this_img, comments="#", delimiter="\t", unpack=False)
    train_X.append(temp_img)


    # # # if using forced 3d
    # # temp_img = np.stack((temp_img,)*3, axis=-1)
    # train_X.append(temp_img)

    # # this was with single tissue sample
    # img_sample_id = os.path.basename(os.path.splitext(this_img)[0])[1:-5]
    # if img_sample_id in metadata_single_tissue['sample_id'].values:
    #     temp_img = np.loadtxt(this_img, comments="#", delimiter="\t", unpack=False)
    #     # temp_img = np.stack((temp_img,)*3, axis=-1)
    #     train_X.append(temp_img)

train_X = np.asarray(train_X)

skf = StratifiedKFold(n_splits=5)

print('Building models')
# dont use these with single value precitors 
AUC_PR = tf.keras.metrics.AUC(curve='PR',name='PR')
AUC_ROC = tf.keras.metrics.AUC(name='ROC')

kfold_counter = 1
for train_index, test_index in skf.split(train_X, train_y):

    base_model = tf.keras.applications.resnet_v2.ResNet50V2(
        include_top=False, weights=None,
        input_shape=(100,100,1)
        )

    last_base_layer = base_model.get_layer('post_bn')
    last_base_output = last_base_layer.output
    x = tf.keras.layers.Flatten()(last_base_output)
    x = tf.keras.layers.Dense(1024,activation = 'relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(1,activation='linear')(x)

    model = Model(inputs = base_model.input,outputs = x)

    model.compile(optimizer=tf.keras.optimizers.Adam(), # learning_rate = 0.1
                # loss=tf.keras.losses.CategoricalCrossentropy(),
                loss=tf.keras.losses.MeanAbsoluteError(),
                # metrics = ['accuracy',AUC_PR,AUC_ROC]
                metrics = ['MeanSquaredError','accuracy']
                )

    checkpoint_path = "training_weights_kfold_" + str(kfold_counter) + "/cp.ckpt" 
    checkpoint_dir = os.path.dirname(checkpoint_path)

    epochs = 1000
    checkpoint = ModelCheckpoint(filepath = checkpoint_path,monitor = "val_loss", mode = 'min',
        save_best_only = True,verbose=1,save_weights_only=True) #use checkpoint instead of sequential() module
    earlystop = EarlyStopping(monitor = 'val_loss', # min_delta=0.00001,
        patience = 100, verbose = 1,restore_best_weights = True) #stop at best epoch
    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor=0.5,
        patience=25, min_lr=0.0001, verbose = 1)

    # first split into training and testing datasets
    kfold_X = train_X[train_index]
    kfold_y = train_y[train_index]
    # this test is not going to be used in the training
    kfold_test = (train_X[test_index],train_y[test_index])

    # # next we split into training and validation
    # val_skf = StratifiedShuffleSplit(n_splits=2) # this only works with n_splits = 2 for some reason but we only need a single split
    # # get the indexes for training and validation
    # train_val_split,_ = val_skf.split(kfold_X, kfold_encoded)
    # # split into training and calidation datasets
    # this_train_X = kfold_X[train_val_split[0]]
    # this_train_y = kfold_y[train_val_split[0]]
    # this_val_X = kfold_X[train_val_split[1]]
    # this_val_y = kfold_y[train_val_split[1]]    

    # fit the model
    history = model.fit(
        # this_train_X,this_train_y,
        kfold_X,kfold_y,
        # validation_data = (this_val_X,this_val_y),
        # validation_split=0.1,
        validation_data = kfold_test,
        epochs=epochs,
        batch_size = 64,
        callbacks= [earlystop, checkpoint,reduce_lr]
    )

    # evaluate results
    eval_result = model.evaluate(kfold_test[0],kfold_test[1],steps=1,return_dict = True,batch_size = 1) #get evaluation results
    res = dict()
    # export evaluation reults to a dict and then print it
    for key in eval_result: res[key] = eval_result[key]

    print("Overall performance")
    print(res)

    plt.figure(figsize = (12,12), dpi = 250)
    for this_key in list(history.history.keys()):
        b = history.history[this_key]
        plt.plot(b,label = this_key)

    plt.legend(loc="upper left")
    plt.ylim([0,5])
    plt.title(json.dumps(res))
    plt.savefig(fname="training_history_" + str(kfold_counter) + ".png")
    # plt.show()

    plt.close('all')

    output = pd.DataFrame()
    for each_tissue in np.unique(sorted_metadata['tissue'].values):

        single_tissue_index = (sorted_metadata['tissue'].values == each_tissue)
        single_tissue_X = train_X[single_tissue_index]
        single_tissue_y = train_y[single_tissue_index]

        eval_result = model.evaluate(single_tissue_X,single_tissue_y,steps=1,return_dict = True,batch_size = 1) #get evaluation results
        eval_result['tissue'] = each_tissue

        output = output.append(eval_result, ignore_index=True)
        output = output.sort_values(by = 'loss',key = natsort_keygen())

    output.to_csv("output_" + str(kfold_counter) + ".csv",index = False)

    kfold_counter = kfold_counter + 1

    del model

print('eof')