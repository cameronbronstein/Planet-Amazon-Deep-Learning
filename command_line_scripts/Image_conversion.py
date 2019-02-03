import sys
import os
import time

import numpy as np
import pandas as pd
import rasterio

from sklearn.model_selection import train_test_split

# random seed
np.random.seed(42)

import warnings
warnings.simplefilter(action='ignore')

# setting file paths
PLANET_KAGGLE_ROOT = os.path.abspath("./kaggle_datasets/")
PLANET_KAGGLE_TRAIN_ROOT = os.path.abspath("./kaggle_datasets/train-tif-v2/")
PLANET_KAGGLE_TEST_ROOT = os.path.abspath("./kaggle_datasets/test-tif-v2/")
PLANET_KAGGLE_LABEL_CSV = os.path.join(PLANET_KAGGLE_ROOT, "./train_v2.csv")
assert os.path.exists(PLANET_KAGGLE_ROOT)
assert os.path.exists(PLANET_KAGGLE_TRAIN_ROOT)
assert os.path.exists(PLANET_KAGGLE_TEST_ROOT)
assert os.path.exists(PLANET_KAGGLE_LABEL_CSV)

# creating a dataframe with the label data
target = pd.read_csv(PLANET_KAGGLE_LABEL_CSV)
target.columns = ["image_name", "labels"]
target.head()

# Build list with unique labels
label_list = []
for tag_str in target['labels'].values:
    label_tags = tag_str.split(' ')
    for label in label_tags:
        if label not in label_list:
            label_list.append(label)

# Add onehot features for every label
for label in label_list:
    target[label] = target['labels'].apply(lambda x: 1 if label in x.split(' ') else 0)

# function to convert the tifs to raster, then a train data numpy array
def convert_to_raster(file_path = PLANET_KAGGLE_TRAIN_ROOT):
    print_counter = 0
    raster_list = []
    for file in range((len(os.listdir(PLANET_KAGGLE_TRAIN_ROOT)) + 1)):
        path = (PLANET_KAGGLE_TRAIN_ROOT + '/train_' + str(file) + '.tif')
        print_counter += 1

        if os.path.exists(path):
            with rasterio.open(path) as src:
                b, g, r, nir = src.read()
                raster = np.dstack([r, g, b, nir])
                raster_list.append(raster)
        
        if file % 1000 == 0:
            print('Current file: {}'.format(path))
            print("{} files have been converted to raster...".format(print_counter))
            print('')
                      
    return np.array(raster_list)

# Converting the training images to arrays.
training_data = convert_to_raster(PLANET_KAGGLE_TRAIN_ROOT)
print("The training data has {} values, each with input shape of {}."\
    .format(training_data.shape[0], training_data.shape[1:]))

np.save("./kaggle_datasets/training_data.npy", arr = training_data)


# setting target data 
target_columns = target.columns[2:]

X_train, X_val, y_train, y_val = train_test_split(training_data,
                                                  target[target_columns]
                                                  )

# Training and Validation Arrays
np.save('./kaggle_datasets/X_train.npy', arr = X_train)
np.save('./kaggle_datasets/X_val.npy', arr = X_val)
np.save('./kaggle_datasets/y_train', arr = y_train)
np.save('./kaggle_datasets/y_val.npy', arr = y_val)

test_data = convert_to_raster(PLANET_KAGGLE_TEST_ROOT)


np.save('./kaggle_datasets/y_val.npy', arr = test_data)