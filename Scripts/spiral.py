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

def f7(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def idx_by_spearman_coef(data,metadata):

    try:
        ages = np.asarray(metadata['Age'].values)
        inital_gene_order = list(data.columns)
    except:
        ages = np.asarray(metadata['age_value'].values)
        inital_gene_order = list(data.columns)[1:]

    output = dict()  

    for count in tqdm(range(len(inital_gene_order))):
        this_gene = inital_gene_order[count]
        these_points = data[this_gene].values
        sprmn_coef = stats.spearmanr(ages,these_points)
        dist_coef = correlation(ages,these_points)
        output[this_gene] = [sprmn_coef.correlation,sprmn_coef.pvalue,dist_coef]
        # output[this_gene] = [1,1,1]

    df = pd.DataFrame.from_dict(output,orient = 'index', columns = ['Spearman_coef','p-value','dist_coef'])
    df = df.sort_values(['Spearman_coef'], ascending = False)

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
num = num_row * num_col # Number of features to be included for analysis, which is also the total number of pixels in image representation
save_image_size = 10 # Size of pictures (in inches) saved during the execution of IGTD algorithm.
max_step = 50000    # The maximum number of iterations to run the IGTD algorithm, if it does not converge.
val_step = 100  # The number of iterations for determining algorithm convergence. If the error reduction rate
                # is smaller than a pre-set threshold for val_step itertions, the algorithm converges.

# # Import the example data and linearly scale each feature so that its minimum and maximum values are 0 and 1, respectively.
print('Loading in data')
data = pd.read_csv('/xdisk/sutphin/GTEx/Normalized/GTEx_merge_L1000.csv',header=0)
data = data.transpose()
metadata = pd.read_csv('/xdisk/sutphin/GTEx/Normalized/GTEx_merge_L1000_sample.csv',header=0)

# sort the tissues by sample id
metadata = metadata.sort_values(by = ['ID'], ascending = True)
data = data.sort_index(ascending = True)
# get rid of metadata that isnt in the data

# this_tissue = 'All_tissues'
this_tissue = 'GTEx_merge_L1000'
# print('min/max transforming')

print('Normalizing data')

idx, df = idx_by_spearman_coef(data,metadata)

data = data.iloc[:,idx]
data = data.iloc[:, :num]
norm_data = min_max_transform(data.values)
norm_data = pd.DataFrame(norm_data, columns=data.columns, index=data.index)