# test data to raster conversion

import sys
import os
import time

import numpy as np
import pandas as pd
import rasterio

import warnings
warnings.simplefilter(action='ignore')

PLANET_KAGGLE_FIXED = os.path.abspath("./kaggle_tif_data/fixed")
PLANET_KAGGLE_TEST_ROOT = os.path.abspath("./kaggle_tif_data/test-tif-v2")
assert os.path.exists(PLANET_KAGGLE_TEST_ROOT)
assert os.path.exists(PLANET_KAGGLE_FIXED)


# convert function

def test_convert_to_raster(
    file_paths = [PLANET_KAGGLE_TEST_ROOT, PLANET_KAGGLE_FIXED],
    file_roots = ['/test_', '/file_']
    ):

    raster_list = []
    print_counter = 0
    for i in range(2):
        print("Converting files from folder: {}".format(file_paths[i]))
        print(len(os.listdir(file_paths[i])) + 1)
        for id_number in range(len(os.listdir(file_paths[i])) + 1):
            path = (file_paths[i] + file_roots[i] + str(id_number) + '.tif')
            print_counter += 1

            if os.path.exists(path):
                with rasterio.open(path) as src:
                    b, g, r, nir = src.read()
                    raster = np.dstack([r, g, b, nir])
                    raster_list.append(raster)
            else:
              print("Incorrect file path: Please troubleshoot.")
              break

            if id_number % 2500 == 0:
                print('Current file: {}'.format(path))
                print("""{} files have been 
                      converted to raster...\n""".format(print_counter))

    return np.array(raster_list)

test_data = test_convert_to_raster()

np.save("./converted_data_files/test_data.npy", arr = test_data)