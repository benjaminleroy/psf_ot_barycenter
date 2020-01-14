# dealing with new data

import numpy as np
import pandas as pd
import scipy.sparse
import sparse
import sklearn
from sklearn.ensemble import RandomForestRegressor
from collections import Counter
import sys, os
import nose
from astropy.io import fits
import re
from PIL import Image
import PIL
import matplotlib.pyplot as plt
import copy
import scipy.ndimage.measurements

import psf_ot_barycenter

data_loc = "data/"
images_loc = data_loc + "hrci_arlac/"




# reading in image ----------------

i_names, i_paths = psf_ot_barycenter.pull_data_galaxy_names(data_path=images_loc,
                                name_grab = lambda x : x.split("_")[-2])

i_names = np.array([np.int(x) for x in i_names])

i_mat_dict = psf_ot_barycenter.pull_data_galaxy(i_names, i_paths,
                    data_path=images_loc)

# reading in summary info --------

summary_file = data_loc + "hrci_arlac_postage.rdb"

with open(summary_file,'r') as f:
    output = f.read()
column_names = output.split("\n")[22].split("\t")


i_summary = pd.read_csv(summary_file, sep = "\s+",
                        skiprows = np.arange(24, dtype = np.int),
                        header=None,
                        names = column_names)


# ...
