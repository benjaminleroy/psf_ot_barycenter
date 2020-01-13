import numpy as np
from collections import Counter
import sys, os
from astropy.io import fits
import re
from PIL import Image
import PIL
import matplotlib.pyplot as plt
import copy
import scipy.ndimage.measurements


def pull_data_galaxy_names(data_path="/Users/benjaminleroy/Dropbox/"+\
                        "DeepLearning_Illustris/",
                        name_grab = lambda x: x.split("_")[-1].split(".")[0]):
    """
    Pull galaxy image names from dropbox directory

    This function reads in data from local dropbox directory, as
    assumes that all ".fits" files in the directory are galaxy images with
    the same format.

    Arguments:
    ----------
    data_path : string
        path to dropbox directory

    Returns:
    --------
    galaxy_names : str vector (n, )
        vector of galaxy names
    galaxy_files : str vector (n, )
        vector of galaxy file names

    """
    file_names = os.listdir(data_path)
    galaxy_files = [x for x in file_names
                        if re.search("fits$",x) is not None]

    galaxy_names = [name_grab(x) for x in galaxy_files]

    assert np.max(list(dict(Counter(galaxy_names)).values())) == 1,\
        "each galaxy should have a unique name"

    return np.array(galaxy_names), np.array(galaxy_files)

def pull_data_galaxy(galaxy_names, galaxy_files,
                    data_path="/Users/benjaminleroy/Dropbox/"+\
                        "DeepLearning_Illustris/"):
    """
    Pull galaxy image data from dropbox directory

    We convert these ".fits" files into a dictionary of matrices, with one
    matrix per galaxy.

    Arguments:
    ----------
    galaxy_names : str vector (n, )
        vector of galaxy names (unique)
    galaxy_files : str vector (n, )
        vector of ".fits" file names within data_path that match up to the
        galaxy_names list ordering
    data_path : string
        path to dropbox directory

    Returns:
    --------
    galaxy_dict : dict
        dictionary of np arrays for each galaxy

    Comments:
    ---------
    A single dictionary cannot contain all your images
    """
    if galaxy_names.shape != galaxy_files.shape:
        raise ValueError("galaxy_names and galaxy_files should be the same "+\
                         "length.")

    if not (np.max(list(dict(Counter(galaxy_names)).values())) == 1):
        raise ValueError("each galaxy_name should be unique.")

    galaxy_dict = dict()


    for name, file_name in zip(galaxy_names, galaxy_files):
        hdul = fits.open(data_path + file_name)
        galaxy_dict[name] = hdul[0].data

    return galaxy_dict

def bounding_box(mat):
    """
    calculates a bounding box around the non-zero values

    Argument:
    ---------
    mat : np.array (n,m)
        image matrix

    Returns:
    --------
    mm_vals : list
        list of min and max value for each axis (row and column)
    range_vals : list
        list of difference between min and max per axis (row and column)
    center : list
        list of image (relative to min/max values) per axis
    center_of_mass : list
        list of center of mass for image
    """
    n, m = mat.shape
    rows = np.arange(n)[mat.sum(axis = 1) != 0]
    cols = np.arange(m)[mat.sum(axis = 0) != 0]

    mm_vals = [[min(rows), max(rows)],
               [min(cols), max(cols)]]
    range_vals = [np.diff(mm_vals[0])[0], np.diff(mm_vals[1])[0]]
    center = [np.mean(mm_vals[0]),
              np.mean(mm_vals[1])]

    r_vals = mat.sum(axis = 1)
    c_vals = mat.sum(axis = 1)
    center_of_mass = list(scipy.ndimage.measurements.center_of_mass(mat))


    return mm_vals, range_vals, center, center_of_mass

def bounding_box_dict(image_dict):
    """
    wrapper for bounding_box for a dictionary (see bounding_box function)

    Arguments:
    ----------
    image_dict : dict
        dictionary of matrix images

    Returns:
    --------
    bb_dict : dict
        dictionary of bounding_box info for matrices in image_dict (same keys)
    """

    bb_dict = dict()
    for name, image in image_dict.items():
        mat = np.array(image)
        bb_dict[name] = bounding_box(mat)

    return bb_dict


def rectify_bounding_box(bb_dict, padding=0):
    """
    expand bounding box for images so they all have the same size bouding box

    Note: if we have an odd sized range, the lower range gets the extra pixel

    Arguments:
    ----------
    bb_dict : dict
        dictionary of bounding_box style info for matrices in image_dict
        (same keys)
    padding : int
        integer for padding on either side (think 1 => you get a padding of 1
        on each side)

    Returns:
    --------
    bb_dict_range : dict
        updated dictionary of bb_dict, changes min,max and range, we preserve
        the "center" even if it's been updated
    """
    # check max range in each dimension
    range_info = np.zeros((0,2))
    for name, bb_info in bb_dict.items():
        range_info = np.concatenate((range_info, np.array([bb_info[1]])))

    max_range_info = np.max(range_info, axis = 0) + 2*padding

    amount_to_add = max_range_info - range_info

    # add amount_to_add to box (but identify when to add 1 more step to bottom)
    range_info_diff = amount_to_add % 2
    amount_to_add = np.floor(amount_to_add/2)

    # then create dictionary containing new ranges
    bb_dict_range = copy.deepcopy(bb_dict)
    for idx, (name, bb_info) in enumerate(bb_dict_range.items()):
        inner_bb = bb_info[0]
        inner_bb[0] = list(np.array(inner_bb[0]) +\
                            np.array([-1,1]) * amount_to_add[idx, 0] +\
                            np.array([-1 * range_info_diff[idx,0],0]))
        inner_bb[1] = list(np.array(inner_bb[1]) +\
                            np.array([-1,1]) * amount_to_add[idx, 1] +\
                            np.array([-1 * range_info_diff[idx,1],0]))

        inner_range = list(np.array(np.diff(np.array(inner_bb)).ravel(),
                                    dtype = np.int))

        bb_dict_range[name] = (inner_bb, inner_range,
                               bb_dict_range[name][2])

    return bb_dict_range


def shrink_single_image(image_mat, bb_info, fill=0):
    """
    shrink a single image to just desired pixels

    Arguments:
    ----------
    image_mat : array (n,m)
        array with full image
    bb_info : tuple
        bounding_box style info for image_mat. We really care that the first
        element contains the range for both dimensions.
    fill : scalar
        if the image isn't as large as bb_info requests this provides the
        value to fill in the image

    Returns:
    --------
    shrunk_image : array (o, p)
        shrunken array
    """
    image_mat = np.array(image_mat)

    n_x, n_y = image_mat.shape


    bb_range_x = np.array(bb_info[0][0], dtype = np.int)
    bb_range_y = np.array(bb_info[0][1], dtype = np.int)

    # for x
    beyond_x = np.array([bb_range_x[0] < 0, bb_range_x[1] >= n_x])
    add_x = beyond_x * np.array([-1 * bb_range_x[0], bb_range_x[1] - n_x + 1])

    bb_range_x[beyond_x] = np.array([0, n_x])[beyond_x]

    # for y
    beyond_y = np.array([bb_range_y[0] < 0, bb_range_y[1] >= n_y])
    add_y = beyond_y * np.array([-1 * bb_range_y[0], bb_range_y[1] - n_y + 1])

    bb_range_y[beyond_y] = np.array([0, n_y])[beyond_y]

    # grab image
    shrunk_image = image_mat[bb_range_x[0]:(bb_range_x[1]+1),:][:,bb_range_y[0]:(bb_range_y[1]+1)]

    # not sure this is correct
    n_new_x, n_new_y = shrunk_image.shape

    shrunk_image = np.concatenate((fill*np.ones((n_new_x, add_y[0])),
                              shrunk_image,
                              fill*np.ones((n_new_x, add_y[1]))), axis = 1)
    n_new_x, n_new_y = shrunk_image.shape
    shrunk_image = np.concatenate((fill*np.ones((add_x[0], n_new_y)),
                              shrunk_image,
                              fill*np.ones((add_x[1], n_new_y))), axis = 0)
    return shrunk_image

def shrink_images(image_dict, bb_dict=None, padding=0, fill=0, rectify=True):
    """
    shrink matrices relative to provided ranges
    needs to be smart about empty values / in corner
    """
    if bb_dict is None:
        bb_dict = bounding_box_dict(image_dict)
    if rectify:
        bb_dict_range = rectify_bounding_box(bb_dict, padding=padding)
    else:
        bb_dict_range = copy.deepcopy(bb_dict)

    new_images = dict()
    for key, image in image_dict.items():
        new_images[key] = shrink_single_image(image,
                                             bb_dict_range[key],
                                             fill=fill)

    return new_images
