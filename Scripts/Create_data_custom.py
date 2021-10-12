import pandas as pd
import numpy as np
import os
from IGTD_Functions import min_max_transform, table_to_image
from sklearn.preprocessing import PolynomialFeatures

def get_factors(x):

    factors_list = []

    for i in range(1, x + 1):
       if x % i == 0:
           factors_list.append(i)

    return factors_list

# Import the example data and linearly scale each feature so that its minimum and maximum values are 0 and 1, respectively.
data = pd.read_csv('../Data/winequality-red.csv',sep=';')
data_labels = data.pop('quality')

p = PolynomialFeatures(degree=3).fit(data)
poly_names = p.get_feature_names(data.columns)
poly_data = p.transform(data)

data = data.iloc[:, :poly_data.shape[1]]
data_names = list(data.columns.values)
norm_data = min_max_transform(poly_data)
norm_data = pd.DataFrame(norm_data, columns=poly_names)

factors_list = get_factors(poly_data.shape[1])

num_row = factors_list[round(len(factors_list)/2)-1]    # Number of pixel rows in image representation
num_col = factors_list[(round(len(factors_list)/2))]    # Number of pixel columns in image representation
num = num_row * num_col # Number of features to be included for analysis, which is also the total number of pixels in image representation
save_image_size = 3 # Size of pictures (in inches) saved during the execution of IGTD algorithm.
max_step = 10000    # The maximum number of iterations to run the IGTD algorithm, if it does not converge.
val_step = 300  # The number of iterations for determining algorithm convergence. If the error reduction rate
                # is smaller than a pre-set threshold for val_step itertions, the algorithm converges.




# Run the IGTD algorithm using (1) the Euclidean distance for calculating pairwise feature distances and pariwise pixel
# distances and (2) the absolute function for evaluating the difference between the feature distance ranking matrix and
# the pixel distance ranking matrix. Save the result in Test_1 folder.
fea_dist_method = 'Euclidean'
image_dist_method = 'Euclidean'
error = 'abs'
result_dir = '../Results/Test_1'
os.makedirs(name=result_dir, exist_ok=True)
table_to_image(norm_data, [num_row, num_col], fea_dist_method, image_dist_method, save_image_size,
               max_step, val_step, result_dir, error)

# Run the IGTD algorithm using (1) the Pearson correlation coefficient for calculating pairwise feature distances,
# (2) the Manhattan distance for calculating pariwise pixel distances, and (3) the square function for evaluating
# the difference between the feature distance ranking matrix and the pixel distance ranking matrix.
# Save the result in Test_2 folder.
fea_dist_method = 'Pearson'
image_dist_method = 'Manhattan'
error = 'squared'
result_dir = '../Results/Test_2'
os.makedirs(name=result_dir, exist_ok=True)
table_to_image(norm_data, [num_row, num_col], fea_dist_method, image_dist_method, save_image_size,
               max_step, val_step, result_dir, error)
