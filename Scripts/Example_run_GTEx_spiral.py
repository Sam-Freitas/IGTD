import pandas as pd
import os
from tqdm import tqdm
from natsort import natsort_keygen
from IGTD_Functions_modified import min_max_transform, table_to_image
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.spatial.distance import correlation, cosine, braycurtis, canberra,euclidean,jensenshannon,mahalanobis,minkowski
import re
from minepy import MINE
import math
import cv2

def spiral(X, Y):
    x = y = 0
    dx = 0
    dy = -1
    out = []

    rx = math.remainder(X,2)
    ry = math.remainder(Y,2)

    if rx == 0:
        x0 = math.floor(X/2) - 1
    else:
        x0 = math.ceil(X/2) - 1

    if ry == 0:
        y0 = math.floor(Y/2) - 1
    else:
        y0 = math.ceil(Y/2) - 1

    for i in range(max(X, Y)**2):
        if (-X/2 < x <= X/2) and (-Y/2 < y <= Y/2):
            out.append([x+x0,y+y0])
            # print([x+x0,y+y0])
            # DO STUFF...
        if x == y or (x < 0 and x == -y) or (x > 0 and x == 1-y):
            dx, dy = -dy, dx
        x, y = x+dx, y+dy
    return out

def spiral_data(expression_data, shape):

    img = np.zeros(shape = shape)

    idx = spiral(shape[0],shape[1])

    if len(shape) < 3:
        for count,this_idx in enumerate(idx):
            base_number = expression_data[count]
            img[this_idx[0],this_idx[1]] = base_number
    else:
        count = 0
        for i,this_idx in enumerate(idx):
            for j in range(shape[2]):
                base_number = expression_data[count]
                img[this_idx[0],this_idx[1],j] = base_number
                count = count + 1

    return img

def idx_by_spearman_coef(data,metadata):

    try:
        ages = np.asarray(metadata['Age'].values)
        inital_gene_order = list(data.columns)
    except:
        ages = np.asarray(metadata['age_value'].values)
        inital_gene_order = list(data.columns)[1:]

    output = dict()  

    mine = MINE(alpha=0.6, c=15, est="mic_approx")

    for count in tqdm(range(len(inital_gene_order))):
        this_gene = inital_gene_order[count]
        these_points = data[this_gene].values

        mine.compute_score(ages,these_points)
        mic = mine.mic()
        tic = mine.tic()

        output[this_gene] = [mic,tic]

    df = pd.DataFrame.from_dict(output,orient = 'index', columns = ['MIC','TIC'])
    df = df.sort_values(['MIC'], ascending = False)

    sorted_gene_order = list(df.index)
    idx = np.zeros(shape = (1,len(sorted_gene_order))).squeeze()

    idx_counter = 0
    for count,this_gene in enumerate(sorted_gene_order):
        idx[idx_counter] = inital_gene_order.index(this_gene)
        idx_counter = idx_counter + 1

    idx = idx.astype(np.int64)

    return idx,df

num_row = 97    # Number of pixel rows in image representation
num_col = 97    # Number of pixel columns in image representation
num = num_row * num_col  # Number of features to be included for analysis, which is also the total number of pixels in image representation
save_image_size = 10 # Size of pictures (in inches) saved during the execution of IGTD algorithm.
max_step = 50000    # The maximum number of iterations to run the IGTD algorithm, if it does not converge.
val_step = 100  # The number of iterations for determining algorithm convergence. If the error reduction rate
                # is smaller than a pre-set threshold for val_step itertions, the algorithm converges.

# # Import the example data and linearly scale each feature so that its minimum and maximum values are 0 and 1, respectively.
print('Loading in data')
data = pd.read_csv('../Data/GTEx_merge_L1000.csv',header=0)
data = data.transpose()
metadata = pd.read_csv('../Data/GTEx_merge_L1000_sample.csv',header=0)

# sort the tissues by sample id
metadata = metadata.sort_values(by = ['ID'], ascending = True)
data = data.sort_index(ascending = True)

idx = metadata['Source'].values == 'Shokhirev'

data = data.iloc[idx,:]
metadata = metadata.iloc[idx,:]

# this_tissue = 'All_tissues'
this_tissue = 'Shokhirev_GTEx_merge_L1000_spiral'
# print('min/max transforming')

print('sorting data')

idx, df = idx_by_spearman_coef(data,metadata)
print('Normalizing data')
data = data.iloc[:,idx]
data = data.iloc[:, :num]
norm_data = min_max_transform(data.values)
norm_data = pd.DataFrame(norm_data, columns=data.columns, index=data.index)

result_dir = '../Results/' + this_tissue + '_gray'
os.makedirs(name=result_dir, exist_ok=True)

print('exporting data')
for i in tqdm(range(norm_data.shape[0])):
    img = spiral_data(norm_data.iloc[i,:], [num_row,num_col])
    img_path = os.path.join(result_dir,norm_data.index[i] + '.png')
    cv2.imwrite(img_path,img*255)

# Run the IGTD algorithm using (1) the Euclidean distance for calculating pairwise feature distances and pariwise pixel
# distances and (2) the absolute function for evaluating the difference between the feature distance ranking matrix and
# the pixel distance ranking matrix. Save the result in Test_1 folder.
# fea_dist_method = 'Euclidean'
# image_dist_method = 'Euclidean'
# error = 'abs'
# result_dir = '../Results/' + this_tissue + '_1_' + str(num)
# os.makedirs(name=result_dir, exist_ok=True)
# table_to_image(data, [num_row, num_col], fea_dist_method, image_dist_method, save_image_size,
#                max_step, val_step, result_dir, error,export_images = True)

# # Run the IGTD algorithm using (1) the Pearson correlation coefficient for calculating pairwise feature distances,
# # (2) the Manhattan distance for calculating pariwise pixel distances, and (3) the square function for evaluating
# # the difference between the feature distance ranking matrix and the pixel distance ranking matrix.
# # Save the result in Test_2 folder.
# fea_dist_method = 'Pearson'
# image_dist_method = 'Manhattan'
# error = 'squared'
# result_dir = '../Results/' + this_tissue + '_2_' + str(num)
# os.makedirs(name=result_dir, exist_ok=True)
# table_to_image(data, [num_row, num_col], fea_dist_method, image_dist_method, save_image_size,
#                max_step, val_step, result_dir, error,export_images = True)
