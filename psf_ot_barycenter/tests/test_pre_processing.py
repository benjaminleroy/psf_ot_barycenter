import numpy as np
import scipy.ndimage.measurements

import psf_ot_barycenter

def test_bounding_box():
    """
    static test for bounding_box function
    """
    a = np.zeros((39,30))
    a[[4,5,6,7,8,9,4,5], [5,6,7,8,9,7,8,5]] = 1

    mm_vals, range_vals, center, center_of_mass = \
        psf_ot_barycenter.bounding_box(a)

    assert mm_vals == [[4,9],[5,9]], \
        "mm_vals calculated incorrectly for static test"

    assert range_vals == [5,4], \
        "range_vals calculated incorrectly for static test"

    assert center == [6.5, 7.0], \
        "center calculated incorrectly for static test"

    assert np.all(center_of_mass == \
                  list(scipy.ndimage.measurements.center_of_mass(a))), \
        "center of mass calculated incorrectly for static test"

def test_bounding_box_dict():
    image_dict = dict()
    a = np.zeros((39,30))
    a[[4,5,6,7,8,9,4,5], [5,6,7,8,9,7,8,5]] = 1
    image_dict["a"] = a
    a_center_of_mass = [6.0, 6.875]

    b = np.zeros((39,30))
    b[[4,2], [12,6]] = 1
    image_dict["b"] = b
    b_center_of_mass = [3.0, 9.0]

    bb_dict = psf_ot_barycenter.bounding_box_dict(image_dict)
    bb_expected_dict = {"a": ([[4,9],[5,9]], [5,4], [6.5, 7.0],
                              a_center_of_mass),
                        "b": ([[2,4],[6,12]], [2,6], [3.0, 9.0],
                              b_center_of_mass)}

    assert bb_dict == bb_expected_dict, \
        "static test returned attributes for bounding_box_dict incorrect"

def test_rectify_bounding_box():
    """
    static tests for rectify_bounding_box
    """
    bb_dict = {"a": ([[4,9],[5,9]], [5,4], [6.5, 7.0]),
               "b": ([[2,4],[6,12]], [2,6], [3.0, 9.0])}

    bb_dict_update = psf_ot_barycenter.rectify_bounding_box(bb_dict)

    for key in bb_dict.keys():
        old = bb_dict[key][0]
        new = bb_dict_update[key][0]

        assert old[0][0] >= new[0][0] and \
               old[1][0] >= new[1][0] and \
               old[0][1] <= new[0][1] and \
               old[1][1] <= new[1][1], \
            "new bounding box should be bigger than old box"

        assert np.diff(new[0]) == bb_dict_update[key][1][0] and \
               np.diff(new[1]) == bb_dict_update[key][1][1], \
            "difference incorrectly updated for bounding box"

        assert bb_dict[key][2] == bb_dict_update[key][2], \
            "rectify bounding box shouldn't update center"

    assert bb_dict_update["a"][1] == bb_dict_update["b"][1], \
        "both items should be of the same size"


def test_shrink_single_image():
    """
    static tests for shrink_single_image
    """

    a = np.zeros((39,30))
    a[[4,5,6,7,8,9,4,5], [5,6,7,8,9,7,8,5]] = 1
    bb_info = ([[4,9],[5,9]], [5,4], [6.5, 7.0])

    a_shrunk = psf_ot_barycenter.shrink_single_image(a, bb_info)

    assert a_shrunk.shape == (6,5),\
        "shrunk image isn't correct size"

    assert np.all(a[4:10,:][:, 5:10] == a_shrunk), \
        "a_shrunk should just be a subset of a"

    # on corner:
    a2 = np.zeros((10,10))
    a2[[4,5,6,7,8,9,4,5], [5,6,7,8,9,7,8,5]] = 1
    bb_info = ([[4,9],[5,9]], [5,4], [6.5, 7.0])
    bb_info2 = ([[4,9],[5,10]], [5,5], [6.5, 7.0])
    bb_info3 = ([[4,10],[5,10]], [6,5], [6.5, 7.0])
    bb_info4 = ([[-1,10],[5,10]], [11,5], [6.5, 7.0])
    bb_info5 = ([[-1,10],[-1,4]], [11,5], [6.5, 7.0])

    # bb_info
    bb = bb_info
    a2_shrunk = psf_ot_barycenter.shrink_single_image(a2, bb)

    assert np.all(a2_shrunk.shape == np.array(bb[1]) + 1),\
        "shrunk image isn't correct size"
    assert np.all(a2[4:10,:][:, 5:10] == a2_shrunk), \
        "a_shrunk should just be a subset of a"

    # bb_info2
    bb = bb_info2
    a2_shrunk = psf_ot_barycenter.shrink_single_image(a2, bb)

    assert np.all(a2_shrunk.shape == np.array(bb[1]) + 1),\
        "shrunk image isn't correct size"
    assert np.all(a2[4:10,:][:, 5:10] == a2_shrunk[:,:-1]), \
        "a2_shrunk should just be a subset of a2"
    assert np.all(a2_shrunk[:,-1] ==0), \
        "added values should fill values (0)"

    # bb_info3
    bb = bb_info3
    a2_shrunk = psf_ot_barycenter.shrink_single_image(a2, bb, fill = -5)

    assert np.all(a2_shrunk.shape == np.array(bb[1]) + 1),\
        "shrunk image isn't correct size"
    assert np.all(a2[4:10,:][:, 5:10] == a2_shrunk[:-1,:-1]), \
        "a2_shrunk should just be a subset of a2"
    assert np.all(a2_shrunk[-1,:] ==-5) and \
           np.all(a2_shrunk[:,-1] ==-5), \
        "added values should fill values (-5)"

    # bb_info4
    bb = bb_info4
    a2_shrunk = psf_ot_barycenter.shrink_single_image(a2, bb, fill = -5)

    assert np.all(a2_shrunk.shape == np.array(bb[1]) + 1),\
        "shrunk image isn't correct size"
    assert np.all(a2[:, 5:10] == a2_shrunk[1:-1,:-1]), \
        "a2_shrunk should just be a subset of a2"
    assert np.all(a2_shrunk[-1,:] ==-5) and \
           np.all(a2_shrunk[:,-1] ==-5) and \
           np.all(a2_shrunk[0,:] == -5), \
        "added values should fill values (-5)"

    # bb_info4
    bb = bb_info5
    a2_shrunk = psf_ot_barycenter.shrink_single_image(a2, bb, fill = -5)

    assert np.all(a2_shrunk.shape == np.array(bb[1]) + 1),\
        "shrunk image isn't correct size"
    assert np.all(a2[:, :5] == a2_shrunk[1:-1,1:]), \
        "a2_shrunk should just be a subset of a2"
    assert np.all(a2_shrunk[-1,:] ==-5) and \
           np.all(a2_shrunk[:,0] ==-5) and \
           np.all(a2_shrunk[0,:] == -5), \
        "added values should fill values (-5)"

def test_shrink_images():
    """
    basic static test for shrink_images
    """
    image_dict = dict()
    a = np.zeros((39,10))
    a[[4,5,6,7,8,9,4,5], [5,6,7,8,9,7,8,5]] = 1
    image_dict["a"] = a

    b = np.zeros((35,30))
    b[[4,2], [12,6]] = 1
    image_dict["b"] = b

    new_images = psf_ot_barycenter.shrink_images(image_dict)

    assert len(new_images) == 2 and \
           type(new_images) == dict, \
        "expected output structure from shrink_images not length 2 dict"

    new_a = new_images["a"]
    new_b = new_images["b"]
    assert new_b.shape == new_a.shape, \
        "shrunk images should now be the same shape"

    new_images2 = psf_ot_barycenter.shrink_images(image_dict, fill = -5)
    new_a = new_images2["a"]
    new_b = new_images2["b"]
    assert np.sum(new_a == -5) > 0 and \
           np.sum(new_b == -5) == 0, \
        "we should expect new_a to get images get filling (near border)"

    new_images3 = psf_ot_barycenter.shrink_images(image_dict,
                                                  padding=1, fill=-5)
    assert np.all(
            np.array(new_images3["a"].shape) - np.array(new_a.shape) == 2), \
        "with 1 unit of padding, we get size increase by 2 in each dimension"

