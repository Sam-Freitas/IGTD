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

    df2 = pd.read_csv('../Data/sorting_csv.csv',header=0, index_col=0)

    # a = list(df.sort_values(['dist_coef'], ascending = False).index)
    # b = list(df.sort_values(['p-value'], ascending = True).index)
    # result = [None]*(len(a)+len(b))
    # result[:50] = a[:75]
    # result[50:] = b[75:]

    # sorted_gene_order = f7(result)

    print('Sorting gene order from L1000')
    sorted_gene_order = list(df.index)
    in_L1000 = list(df2.index)

    idx = np.zeros(shape = (1,len(sorted_gene_order))).squeeze()

    idx_counter = 0
    for count,this_gene in enumerate(sorted_gene_order):
        if this_gene in in_L1000:
            idx[idx_counter] = inital_gene_order.index(this_gene)
            idx_counter = idx_counter + 1

    idx = idx.astype(np.int64)

    return idx,df

def add_2_features(data):

    new = []

    for i in range(data.shape[0]):
        new.append([np.std(data.iloc[i,:]),np.mean(data.iloc[i,:])])

    new = np.asarray(new)

    data['std'] = new[:,0]
    data['mean'] = new[:,1]

    return data

# num_row = 74    # Number of pixel rows in image representation
# num_col = 130    # Number of pixel columns in image representation
# num = num_row * num_col # Number of features to be included for analysis, which is also the total number of pixels in image representation
# save_image_size = 10 # Size of pictures (in inches) saved during the execution of IGTD algorithm.
# max_step = 20000    # The maximum number of iterations to run the IGTD algorithm, if it does not converge.
# val_step = 1000  # The number of iterations for determining algorithm convergence. If the error reduction rate
#                 # is smaller than a pre-set threshold for val_step itertions, the algorithm converges.

num_row = 96    # Number of pixel rows in image representation
num_col = 100    # Number of pixel columns in image representation
num = num_row * num_col # Number of features to be included for analysis, which is also the total number of pixels in image representation
save_image_size = 10 # Size of pictures (in inches) saved during the execution of IGTD algorithm.
max_step = 100    # The maximum number of iterations to run the IGTD algorithm, if it does not converge.
val_step = 1000  # The number of iterations for determining algorithm convergence. If the error reduction rate
                # is smaller than a pre-set threshold for val_step itertions, the algorithm converges.

# # Import the example data and linearly scale each feature so that its minimum and maximum values are 0 and 1, respectively.
print('Loading in data')
# data = pd.read_csv('../Data/normalized_training_data_rot.csv',header=0, index_col=0)
# metadata = pd.read_csv('../Data/human_data/meta_filtered.csv',header=0, index_col=0)
# # sort data 
# metadata = metadata.sort_values(by = ['SRR.ID'], ascending = True)
# data = data.sort_index(axis = 0, ascending = True)

# # this_tissue = 'All_tissues'
# this_tissue = 'Liver;liver hepatocytes'
# healthy_index = metadata['Healthy'].values == True
# tissue_index = metadata['Tissue'].values == this_tissue

# data_index = healthy_index*tissue_index

# data = data.iloc[data_index,:]
# metadata = metadata.iloc[data_index,:]
# sprt_idx,sort_help = idx_by_spearman_coef(data,metadata)
# # data_std = data.std()
# # sorted_std_idx_ascend = np.argsort(data_std.values)
# # data = data.iloc[:, sorted_std_idx_ascend[-num:]]
# data = data.iloc[:, sprt_idx[:num]]
# data = add_2_features(data)

data = pd.read_csv('/xdisk/sutphin/GTEx/RNA-seq-data/whole_blood_counts_rot.csv',header=0,skiprows=2)
metadata = pd.read_csv('/xdisk/sutphin/GTEx/tsvs/subject.csv',header=0)

# shorted the data descriptions into tissue samples
for count in range(data.shape[0]):
    temp = data.iloc[count,0]
    idx = [m.start() for m in re.finditer('-',temp)]
    temp = temp[0:idx[1]]
    data.iloc[count,0] = temp
# sort the tissues by sample id
metadata = metadata.sort_values(by = ['source_subject_id'], ascending = True)
data = data.sort_values(by = ['Description'], ascending = True)
# get rid of metadata that isnt in the data
list_meta_ssid = list(metadata['source_subject_id'].values)
idx_in_dataset = []
for count in range(data.shape[0]):
    this_data_name = data.iloc[count,0]
    idx = list_meta_ssid.index(this_data_name)
    idx_in_dataset.append(idx)

metadata = metadata.iloc[idx_in_dataset,:]

print('checking shape')
assert metadata.shape[0] == data.shape[0]

# this_tissue = 'All_tissues'
this_tissue = 'test'
sprt_idx,sort_help = idx_by_spearman_coef(data,metadata)
# data_std = data.std()
# sorted_std_idx_ascend = np.argsort(data_std.values)
# data = data.iloc[:, sorted_std_idx_ascend[-num:]]
data = data.iloc[:, sprt_idx[:num]]
# print('min/max transforming')
for count in tqdm(range(data.shape[0])):
    temp_data = np.clip(data.iloc[count,:].values,0,1*(10*6))
    data.iloc[count,:] = temp_data/np.max(temp_data)
# data = add_2_features(data)
# norm_data = min_max_transform(data.values)
# norm_data = pd.DataFrame(norm_data, columns=data.columns, index=data.index)

# Run the IGTD algorithm using (1) the Euclidean distance for calculating pairwise feature distances and pariwise pixel
# distances and (2) the absolute function for evaluating the difference between the feature distance ranking matrix and
# the pixel distance ranking matrix. Save the result in Test_1 folder.
fea_dist_method = 'Euclidean'
image_dist_method = 'Euclidean'
error = 'abs'
result_dir = '../Results/' + this_tissue + '_1_' + str(num)
os.makedirs(name=result_dir, exist_ok=True)
table_to_image(data, [num_row, num_col], fea_dist_method, image_dist_method, save_image_size,
               max_step, val_step, result_dir, error,export_images = True)

# Run the IGTD algorithm using (1) the Pearson correlation coefficient for calculating pairwise feature distances,
# (2) the Manhattan distance for calculating pariwise pixel distances, and (3) the square function for evaluating
# the difference between the feature distance ranking matrix and the pixel distance ranking matrix.
# Save the result in Test_2 folder.
fea_dist_method = 'Pearson'
image_dist_method = 'Manhattan'
error = 'squared'
result_dir = '../Results/' + this_tissue + '_2_' + str(num)
os.makedirs(name=result_dir, exist_ok=True)
table_to_image(data, [num_row, num_col], fea_dist_method, image_dist_method, save_image_size,
               max_step, val_step, result_dir, error,export_images = True)
