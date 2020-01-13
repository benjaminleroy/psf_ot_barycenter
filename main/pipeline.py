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


# analysis ------------------------


data_path = "data/HRCI_ARLac_old/"


#
# pulling data --------------------
#

# reading in image ----------------

i_names, i_paths = psf_ot_barycenter.pull_data_galaxy_names(data_path=data_path,
                                name_grab = lambda x : x.split("_")[-2])

i_paths = i_paths[i_names != 'ARLac']
i_names = i_names[i_names != 'ARLac']

i_names = np.array([np.int(x) for x in i_names])

i_mat_dict = psf_ot_barycenter.pull_data_galaxy(i_names, i_paths,
                    data_path=data_path)

# reading in summary info --------

with open(data_path+"HRCI_ARLac_summary.txt",'r') as f:
    output = f.read()
column_names = output.split("\n")[0].split(",")


i_summary = pd.read_csv(data_path+"HRCI_ARLac_summary.txt", sep = "\s+",
                        skiprows = [0], header=None,
                        names = column_names)


# rotating images -----------------

i_image_dict = dict()
for name, mat in i_mat_dict.items():
    i_image = Image.fromarray(mat)
    angle = i_summary[i_summary.ObsID == name][["Roll [deg]"]].values[0][0]
    i_rotated = i_image.rotate(angle)
    i_image_dict[name] = i_rotated


# EDA
# summation visuals

from functools import reduce
addition_pre = reduce((lambda x, y: np.array(x) + np.array(y)),
                      i_mat_dict.values())
addition = reduce((lambda x, y: np.array(x) + np.array(y)),
                  i_image_dict.values())

fig, ax = plt.subplots(ncols = 2, nrows = 1)
ax[0].imshow(addition_pre, cmap = "binary")
ax[0].set_title("Sum pre rotation")
ax[1].imshow(addition, cmap = "binary")
ax[1].set_title("Sum with rotation")

plt.savefig(fname = "images/additional_rotation_vis.png")
plt.close()

fig, ax = plt.subplots(ncols = 2, nrows = 1)
ax[0].imshow(np.log(1+addition_pre), cmap = "binary")
ax[0].set_title("Sum pre rotation")
ax[1].imshow(np.log(1+addition), cmap = "binary")
ax[1].set_title("Sum with rotation")
fig.suptitle("log(1+x) tranformed")

plt.savefig(fname = "images/additional_rotation_vis_log.png")
plt.close()


bb_pre = psf_ot_barycenter.bounding_box_dict(i_mat_dict)
bb_rotate = psf_ot_barycenter.bounding_box_dict(i_image_dict)

centers = np.zeros((0,5))
for key in bb_pre.keys():
    centers = np.concatenate((centers,
                              np.array([[key] + bb_pre[key][3] +\
                                       bb_rotate[key][3]])),
                             axis = 0)

centers_df = pd.DataFrame(centers, columns = ["id", "x", "y", "x rotate", "y rotate"])

centers_df.to_csv("images/centers_rotation.csv")
# single image example

if False:
    for number in [1294, 13182]:
        my_image = i_mat_dict[number]
        fig, ax = plt.subplots(ncols = 2, nrows = 2)
        ax[1,1].imshow(my_image, cmap = "binary", aspect = "auto")
        ax[1,0].barh(np.arange(1024), np.log(1+my_image.sum(axis = 1)), color = "black")
        ax[0,1].bar(np.arange(1024), np.log(1+my_image.sum(axis = 0)), color = "black")
        ax[1,0].invert_yaxis()
        ax[1,0].invert_xaxis()
        #fig.suptitle("Image 1294")
        ax[1,0].set_xlabel("log(1+sum)")
        ax[0,1].set_ylabel("log(1+sum)")

        my_counts = Counter(my_image.ravel())
        levels = np.array(list(dict(my_counts).keys()))
        vals = np.array(list(dict(my_counts).values()))

        ax[0,0].bar(x = levels, height = vals)
        ax[0,0].set_ylabel("count (on a log scale)")
        ax[0,0].set_yscale("log")
        ax[0,0].set_xlabel("value in pixel")
        #ax[0,0].remove()
        fig.tight_layout()

        plt.savefig(fname = "images/noisy_image_example"+str(number)+".png")
        plt.close()




# distribution of counts across all images
if False:
    all_counts = Counter([])
    for key, item in i_image_dict.items():
        all_counts += Counter(np.array(item).ravel())

    levels = np.array(list(dict(all_counts).keys()))
    vals = np.array(list(dict(all_counts).values()))

    plt.bar(x = levels, height = np.log(vals))
    plt.ylabel("log(count)")
    plt.title("Value in box vs number observed in all images")



# s_image_dict = shrink_images(i_image_dict, padding=0, fill=0)


# center of mass weights
centers_of_mass = np.zeros((0,2))
cm_names = []
for key in bb_rotate.keys():
    centers_of_mass = np.concatenate((centers_of_mass,
                              np.array([bb_rotate[key][3]])),
                             axis = 0)
    cm_names.append(key)

loc = np.array([1024/2]*2)

from cvxopt import matrix, solvers

def run_lp(centers_of_mass, loc):
    """
    estimate weights for the following function

    sum w_i d_i

    s.t.
        w_i >= 0 for all i

        sum_i w_i = 1
        sum_i w_i x_i = x_loc
        sum_i w_i y_i = y_loc

    Arguments:
    ----------
    centers_of_mass : array (n, 2)
    loc : array (, 2)


    Returns:
    --------
    weights : array (n, )

    """

    # calculate distance

    dist = np.sqrt(np.sum((centers_of_mass - loc)**2, axis = 1))
    c = matrix(dist)

    # equal
    A = matrix(np.concatenate((np.ones(dist.shape[0]).reshape((1,-1)),
                       centers_of_mass.T), axis = 0))

    b = matrix(np.array([1] + list(loc)))


    # greater than
    G = matrix(-np.eye(dist.shape[0])) # negative for the correct >=
    h = matrix(np.zeros(dist.shape[0]))

    # options = dict()
    # options["show_progress"] = False

    sol = solvers.lp(c = c, G = G, h = h, A = A, b = b)

    weights = sol["x"]

    return weights


weights = run_lp(centers_of_mass, loc)
centers_df2 = centers_df.copy()
centers_df2["weights"] = weights
centers_df2.to_csv("images/centers_rotation_weights.csv")

# ####
# import rpy2
# import rpy2.robjects as robjects
# import rpy2.robjects.packages as rpackages

# utils = rpackages.importr('utils')
# utils.chooseCRANmirror(ind=1) # select the first mirror in the list


# # R package names
# packnames = ['depth']

# # R vector of strings
# from rpy2.robjects.vectors import StrVector

# # Selectively install what needs to be install.
# # We are fancy, just because we can.
# names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
# if len(names_to_install) > 0:
#     utils.install_packages(StrVector(names_to_install))

for key, image in i_image_dict.items():
    np.savetxt(fname="images/"+str(key)+"rotated.csv",
               X=np.array(image), delimiter=",")


# a different way to calculate the boxes
depth_based_boxes_df = pd.read_csv("images/boxes_range.csv")



def min_max_to_bb_dict(dbb_df):
    """
    converts data frame with min/max info to bb_dict
    """
    keys = dbb_df["image"].unique()
    bb_dict = dict()
    for key in keys:
        inner_df = dbb_df.loc[dbb_df["image"] == key]

        r_min = inner_df.loc[(inner_df["cat"] == "min") & (inner_df["dim"] == "rows")]["values"].values[0]
        r_max = inner_df.loc[(inner_df["cat"] == "max") & (inner_df["dim"] == "rows")]["values"].values[0]
        c_min = inner_df.loc[(inner_df["cat"] == "min") & (inner_df["dim"] == "columns")]["values"].values[0]
        c_max = inner_df.loc[(inner_df["cat"] == "max") & (inner_df["dim"] == "columns")]["values"].values[0]

        r_center = (r_max + r_min)/2
        c_center = (c_max + c_min)/2

        r_range = r_max - r_min
        c_range = c_max - c_min

        out = ([[r_min, r_max], [c_min, c_max]], [r_range, c_range], [r_center, c_center])

        bb_dict[key] = out
    return bb_dict


my_bb_dict = psf_ot_barycenter.min_max_to_bb_dict(depth_based_boxes_df)

my_bb_dict2 = psf_ot_barycenter.rectify_bounding_box(my_bb_dict)

sliced_images_original = psf_ot_barycenter.shrink_images(i_image_dict, bb_dict=my_bb_dict, rectify = False)

sliced_images = psf_ot_barycenter.shrink_images(i_image_dict, bb_dict=my_bb_dict2, rectify = False)

# visualize these sliced images vs the true images

# maybe just look over images and draw new bounding box?

# def vis(image_dict):
#     n = len(image_dict)
#     fig, ax = plt.subplots(nrows = np.floor(n/2),
#                            ncol = np.ceiling(n/2))

from matplotlib import collections as mc

for key in i_image_dict.keys():
    full_image = i_image_dict[key]
    sliced_o = sliced_images_original[key]
    sliced_image = sliced_images[key]

    bb_box = my_bb_dict2[key][0]
    lines = [ [(bb_box[1][0], bb_box[0][0]), (bb_box[1][1], bb_box[0][0])],
              [(bb_box[1][0], bb_box[0][1]), (bb_box[1][1], bb_box[0][1])],
              [(bb_box[1][0], bb_box[0][0]), (bb_box[1][0], bb_box[0][1])],
              [(bb_box[1][1], bb_box[0][0]), (bb_box[1][1], bb_box[0][1])]

        ]

    # boxed
    fig, ax = plt.subplots()
    ax.imshow(full_image, cmap = "binary")
    lc = mc.LineCollection(lines)
    ax.add_collection(lc)
    ax.set_title(key)

    plt.savefig("images/" + str(key) + "boxed_image.png")
    plt.close()

    # zoomed_not_standard
    fig, ax = plt.subplots()
    ax.imshow(sliced_o, cmap = "binary")
    ax.set_title(str(key)+", sliced (personal)")

    plt.savefig("images/" + str(key) + "zoomed_not_standard_image.png")
    plt.close()

    # zoomed
    fig, ax = plt.subplots()
    ax.imshow(sliced_image, cmap = "binary")
    ax.set_title(str(key)+", sliced")

    plt.savefig("images/" + str(key) + "zoomed_image.png")
    plt.close()


    # boxed_log
    fig, ax = plt.subplots()
    ax.imshow(np.log(np.array(full_image) +1), cmap = "binary")
    lc = mc.LineCollection(lines)
    ax.add_collection(lc)
    ax.set_title(str(key) + " (log(x+1))")

    plt.savefig("images/" + str(key) + "log_boxed_image.png")
    plt.close()

    # zoomed_not_standard
    fig, ax = plt.subplots()
    ax.imshow(np.log(np.array(sliced_o)+1), cmap = "binary")
    ax.set_title(str(key)+", sliced (personal)")

    plt.savefig("images/" + str(key) + "log_zoomed_not_standard_image.png")
    plt.close()

    # zoomed_log
    fig, ax = plt.subplots()
    ax.imshow(np.log(np.array(sliced_image) +1), cmap = "binary")
    ax.set_title(str(key)+", sliced (log(x+1))")

    plt.savefig("images/" + str(key) + "log_zoomed_image.png")
    plt.close()


# berycenters
import ot

weights_all = np.array(centers_df2.weights)
selected_idx = np.array(centers_df2.id, dtype = np.int)[weights_all > .01]
selected_weights = weights_all[weights_all > .01]

A = np.array([sliced_images[idx]/sliced_images[idx].sum()
                     for idx in selected_idx])

def correct_shape_grow(A):
    """
    A is 3d
    """
    _, n, m = A.shape
    if (n != m):
        diff = np.abs(m - n)
        size = (A.shape[0], diff*(m > n) + n *(n > m),
                            diff*(n > m) + m *(m > n))
        A = np.concatenate((A, np.zeros(size)), axis = 1*(m > n) + 2*(n > m))

    return A, (n, m)

A2, original_size = correct_shape_grow(A)


reg = .002
weights = selected_weights/selected_weights.sum()

bcenter2 = ot.bregman.convolutional_barycenter2d(A2, reg, weights)

def correct_shape_shrink(b, size):
    """
    b is 2d
    """
    n, m = size
    diff = np.abs(n - m)
    if n == m:
        return b
    elif n < m:
        b2 = b[:n,:]
    else:
        b2 = b[:,:m]

    return b2


bcenter = correct_shape_shrink(bcenter2, original_size)

# if (n != m):
#     if (n > m):
#     bcenter2 = bcenter[]


fig, ax = plt.subplots(nrows = 2, ncols = 2)
ax[1,1].imshow(bcenter, cmap = "binary")
ax[1,1].set_title("berycenter")
ax[0,1].imshow(A[0,:,:], cmap = "binary")
ax[0,1].set_title("weight:"+str(np.round(weights[0],3)))
ax[0,0].imshow(A[1,:,:], cmap = "binary")
ax[0,0].set_title("weight:"+str(np.round(weights[1],3)))
ax[1,0].imshow(A[2,:,:], cmap = "binary")
ax[1,0].set_title("weight:"+str(np.round(weights[2],3)))
fig.suptitle("actual images vs smoothed berycenter")
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("images/berycenter_example_actual.png")
plt.close()



smoothed_image = dict()
for idx in np.arange(3, dtype = np.int):
    weights_inner = np.zeros(3)
    weights_inner[idx] = 1
    b2 = ot.bregman.convolutional_barycenter2d(A2, reg, weights_inner)
    smoothed_image[idx] = correct_shape_shrink(b2, original_size)


fig, ax = plt.subplots(nrows = 2, ncols = 2)
ax[1,1].imshow(bcenter, cmap = "binary")
ax[1,1].set_title("berycenter")
ax[0,1].imshow(smoothed_image[0], cmap = "binary")
ax[0,1].set_title("weight:"+str(np.round(weights[0],3)))
ax[0,0].imshow(smoothed_image[1], cmap = "binary")
ax[0,0].set_title("weight:"+str(np.round(weights[1],3)))
ax[1,0].imshow(smoothed_image[2], cmap = "binary")
ax[1,0].set_title("weight:"+str(np.round(weights[2],3)))
fig.suptitle("smoothed images vs smoothed berycenter")
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("images/berycenter_example.png")
plt.close()


# lp attept (resolution 32 x 32)

# want: 1024 total pixels
# have 71 * 105 = 7455
# change to 71/np.sqrt(7); 105/np.sqrt(7)


dim_size = 26, 39
A_start_small = [np.array(Image.fromarray(x).resize((dim_size[1], dim_size[0]),
                                                    resample = PIL.Image.BOX))
                 for x in A]

A3_start = A_start_small
A3 = np.array(A_start_small)

A3_ravel = A3.reshape((3,-1)).T

def calc_distance(n,m):
    """
    calculate distance matrix for a n x m image (euclidean)

    this returns the squared distance

    a = array([[ 0,  1,  2,  3],
               [ 4,  5,  6,  7],
               [ 8,  9, 10, 11]])
    a.ravel() =  array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])

    calc_distance(3,4) =
        array([[ 0.,  1.,  4.,  9.,  1.,  2.,  5., 10.,  4.,  5.,  8., 13.],
               [ 1.,  0.,  1.,  4.,  2.,  1.,  2.,  5.,  5.,  4.,  5.,  8.],
               [ 4.,  1.,  0.,  1.,  5.,  2.,  1.,  2.,  8.,  5.,  4.,  5.],
               [ 9.,  4.,  1.,  0., 10.,  5.,  2.,  1., 13.,  8.,  5.,  4.],
               [ 1.,  2.,  5., 10.,  0., ...
    """
    diff_first = np.concatenate([np.diag(v = i**2 * np.ones(n-i), k = i).reshape((1,n,n))
                for i in np.arange(n, dtype = np.int)],axis = 0).sum(axis = 0)
    diff_first += diff_first.T

    diff_second = np.concatenate([np.diag(v = i**2 * np.ones(m-i), k = i).reshape((1,m,m))
                for i in np.arange(m, dtype = np.int)],axis = 0).sum(axis = 0)
    diff_second += diff_second.T

    diff_all2 = np.add.outer(diff_first, diff_second)
    diff_all2_final = np.zeros((n*m, n*m))

    row_start = 0
    for row_idx in range(n):
        col_start = 0
        for col_idx in range(n):
            diff_all2_final[row_start:(row_start+m),col_start:(col_start+m)] = diff_all2[row_idx,col_idx]
            col_start = col_start + m
        row_start = row_start + m

    return diff_all2_final


def calc_distance_2(n,m):
    np.add.outer(a, b)#np.einsum("ij,kl")

M = calc_distance(dim_size[0],dim_size[1])

raveled_bc, log = ot.lp.barycenter(A3_ravel, M, weights = weights, verbose=False,log=True)


if False:
    weights_test = np.array([0,0,1])
    raveled_bc_test, log_test = ot.lp.barycenter(A3_ravel, M, weights = weights_test, verbose=False,log=True)
    bc_test = raveled_bc_test.T.reshape(dim_size, order="F" ).T

bc = raveled_bc.T.reshape(dim_size, order="C")
vis1 = A3[0,:].T.reshape(dim_size, order="F").T
np.savetxt(X = bc, fname = "images/lp1014_berycenter.txt")
np.savetxt(X = raveled_bc, fname = "images/lp1014_berycenter_raveled.txt")


fig, ax = plt.subplots(nrows = 2, ncols = 2)
ax[1,1].imshow(bc, cmap = "binary")
ax[1,1].set_title("berycenter")
ax[0,1].imshow(A3_start[0], cmap = "binary")
ax[0,1].set_title("weight:"+str(np.round(weights[0],3)))
ax[0,0].imshow(A3_start[1], cmap = "binary")
ax[0,0].set_title("weight:"+str(np.round(weights[1],3)))
ax[1,0].imshow(A3_start[2], cmap = "binary")
ax[1,0].set_title("weight:"+str(np.round(weights[2],3)))
fig.suptitle("pixelated images vs pixelated berycenter")
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("images/berycenter_example_lp1014.png")
plt.close()




# only between 2 images:

dim_size = 26, 39
A_start_small = [np.array(Image.fromarray(x).resize(dim_size,
                                                    resample = PIL.Image.BOX))
                 for x in A]

A4_start = [A_start_small[0], A_start_small[1]]
A4 = np.array(A4_start)

A4_ravel = A4.reshape((2,-1)).T

M = calc_distance(dim_size[0],dim_size[1])

weights4 = np.array([.5,.5])

raveled_bc4, log4 = ot.lp.barycenter(A4_ravel, M, weights = weights4, verbose=False,log=True)


bc4 = raveled_bc4.T.reshape(dim_size, order="C")
np.savetxt(X = bc4, fname = "images/lp1014_berycenter_2test.txt")
np.savetxt(X = raveled_bc4, fname = "images/lp1014_berycenter_raveled_2test.txt")


fig, ax = plt.subplots(nrows = 2, ncols = 2)
ax[1,1].imshow(bc, cmap = "binary")
ax[1,1].set_title("berycenter")
ax[0,1].imshow(A4_start[0], cmap = "binary")
ax[0,1].set_title("weight:"+str(np.round(weights4[0],3)))
ax[0,0].imshow(A4_start[1], cmap = "binary")
ax[0,0].set_title("weight:"+str(np.round(weights4[1],3)))
fig.suptitle("pixelated images vs pixelated berycenter")
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("images/berycenter_example_lp1014_2test.png")
plt.close()


# between 2 gaussians


xedges = np.linspace(-4,2,20)
yedges = np.linspace(-2,2,10)
x = np.random.normal(0,1,3000)
y = np.random.normal(0,1,3000)
image1, _, _ = np.histogram2d(x, y, bins=(xedges, yedges))
image1 = image1.T

x2 = np.random.normal(-2,1,3000)
y2 = np.random.normal(0,1,3000)

image2, _, _ = np.histogram2d(x2, y2, bins=(xedges, yedges))
image2 = image2.T

dim_size_gaus = image1.shape
M_gaus = calc_distance(dim_size_gaus[0],dim_size_gaus[1])

weights_gaus = np.array([.5,.5])

A_start_small_gaus = [np.array(Image.fromarray(x).resize((dim_size_gaus[1], dim_size_gaus[0]),
                                                    resample = PIL.Image.BOX))
                 for x in [image1, image2]]

#A_start_small_gaus2 = [image1, image2]

A_start_gaus = [A_start_small_gaus[0], A_start_small_gaus[1]]
A_gaus = np.array(A_start_gaus)

A_gaus_ravel = A_gaus.reshape((2,-1)).T

weights_gaus = np.array([.5,.5])

raveled_bc_gaus, log_gaus = ot.lp.barycenter(A_gaus_ravel, M_gaus,
                                             weights = weights_gaus,
                                             verbose=False,log=True)

bc_gaus = raveled_bc_gaus.T.reshape(dim_size_gaus, order="C")
np.savetxt(X = bc_gaus, fname = "images/gaus_berycenter_2test.txt")
np.savetxt(X = raveled_bc_gaus, fname = "images/gaus_berycenter_raveled_2test.txt")

fig, ax = plt.subplots(nrows = 2, ncols = 2)
ax[1,1].imshow(bc_gaus, cmap = "binary")
ax[1,1].set_title("berycenter")
ax[0,1].imshow(image1, cmap = "binary")
ax[0,0].imshow(image2, cmap = "binary")
fig.suptitle("pixelated gaussian vs lp berycenter")
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("images/berycenter_example_gaus_2test.png")
plt.close()

