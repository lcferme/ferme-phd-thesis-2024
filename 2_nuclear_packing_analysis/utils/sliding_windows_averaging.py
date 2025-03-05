"""
@authors:   Lucrezia Ferme @ Norden group @ IGC
            Allyson Quinn Ryan @ Modes group @ MPI-CBG and @ Haase group @ TU Dresden
@descript:  Functions for the extraction of several nuclear features to be used
            with regionprops.
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from tifffile import imread, imwrite
import itertools
import math
from PIL import Image
from PIL.TiffTags import TAGS
import matplotlib.pyplot as plt
from skimage.measure import regionprops
from skimage.segmentation import clear_border, relabel_sequential
from skimage.morphology import remove_small_objects
import pyclesperanto_prototype as cle
from preprocess import get_spatial_calib_tiff, clearing
from features import unit_vector, angle_between

# ---------
def stopwatch(text : str = None):
    if not stopwatch.verbose:
        return
    import time

    if text is not None:
        print("-------------------------> " + text + " took " + str(int((time.time() - stopwatch.timestamp) * 1000)) + "ms")

    stopwatch.timestamp = time.time()

stopwatch.timestamp = 0
stopwatch.verbose = True

# ---------

# defining the window size and shift step
def define_window_size(z_res, y_res, x_res, size_micron): #size micron = 10 originally, 20 for bigger, 5 for smaller
    """ Define the size step in pixels based on the chosen size of the window size (in micron)
    """
    z_step = math.floor(size_micron/z_res)
    y_step = math.floor(size_micron/y_res)
    x_step = math.floor(size_micron/x_res)
    return (z_step, y_step, x_step)

def define_shift_step(window_size, shift=0.20):
    """ Define the shift step accounting for a minimum overlap of windows, given as 'shift'
    """
    shift_step = np.rint(np.asarray(window_size)*shift)
    return shift_step.astype(np.uint8)

def test_number_of_steps_per_image(img_size, window_size, shift_step):
    window_size = np.asarray(window_size)
    n_steps = np.rint(np.asarray(img_size)/window_size)
    max_index_steps = (n_steps*window_size)+shift_step
    return max_index_steps
# ---------

def sliding_window(img, window_size, shift_step):

    #det sizes of window panes
    arr_size = img.shape
    z_size = arr_size[0]
    y_size = arr_size[1]
    x_size = arr_size[2]

    #build window sets
    z_windows = tuple([(i, i+window_size[0]) for i in range(0, z_size, shift_step[0])])
    y_windows = tuple([(i, i+window_size[1]) for i in range(0, y_size, shift_step[1])])
    x_windows = tuple([(i, i+window_size[2]) for i in range(0, x_size, shift_step[2])])

    window_comps = list(itertools.product(z_windows, y_windows, x_windows))
    return window_comps
# ---------

def get_touch_matrix(img, dilation):
    """ Takes a labelled image with n labels and an integer number for expanding each labelled object.
    Returns an (n+1)*(n+1) matrix filled with 0s and each row and column correspond to one label. Where object i and object j
    touch, the position matrix[i][j] is equal to 1. Returns a list of number of neighbours per labelled object.
    """

    assert isinstance(dilation, int), "The dilation value should be an integer"
    # extend labels by a dilation factor
    img_gpu = cle.push_zyx(img)
    extended_labels = cle.extend_labels_with_maximum_radius(img_gpu, radius = dilation)

    # get matrix of touching objects
    touch_matrix = cle.generate_touch_matrix(extended_labels)
    cle.set_column(touch_matrix, 0, 0)
    neighbors = cle.count_touching_neighbors(touch_matrix)[0]


    return cle.pull(touch_matrix), cle.pull(neighbors)


# ------------------------------------
####################################################################################################################################
###                                                  FUNCTIONS OF INTEREST FOR AVERAGING
#####################################################################################################################################

def count_nuclei_per_window(ids):
    return len(ids)

# ------------------------------------

def count_number_of_contacts_per_window(label, touch_matrix):
    row = touch_matrix[label] #ids[j]
    neighbors = np.where(row)
    n_neighbors = len(neighbors[0])
    return n_neighbors

# ------------------------------------

def get_average_angle_between_nematics_per_window(j, ids, touch_matrix, primary_nematic, secondary_nematic):
    angles_prim, angles_sec = [], []
    
    # iterate through the other ids in the window and calculate angle to present ids[j]
    for label in ids[j:]:
        if ids[j] != label:
            n = label -1 # -1 because of the way I saved the arrays for each label
            prim_nem_a = abs(primary_nematic[0][ids[j]-1]-primary_nematic[1][ids[j]-1])
            prim_nem_b = abs(primary_nematic[0][n]-primary_nematic[1][n])
            sec_nem_a = abs(secondary_nematic[0][ids[j]-1]-secondary_nematic[1][ids[j]-1])
            sec_nem_b = abs(secondary_nematic[0][n]-secondary_nematic[1][n])

            angles = (angle_between(prim_nem_a, prim_nem_b), angle_between(sec_nem_a, sec_nem_b))
          
            if (not math.isnan(angles[0]) and not math.isnan(angles[1])):
                angles_prim.append(angles[0])
                angles_sec.append(angles[1])

    return (angles_prim, angles_sec)

# ------------------------------------

def get_angles_between_director_and_nematics_per_window(directors):
    angles_to_director_list = []
    avg_director = np.add.reduce(directors)/directors.shape[0]
    #avg_director_hat = avg_director/np.linalg.norm(avg_director)

    for i in range(len(directors)):
        nematic_director = directors[i]
        angles_to_director = angle_between(nematic_director, avg_director)
          
        if not math.isnan(angles_to_director) :
            angles_to_director_list.append(angles_to_director)
          
    return angles_to_director_list

# ------------------------------------

def order_parameter_s(vectors):
    #average direction
    #print('error', np.add.reduce(vectors), vectors.shape)
    avg_v = np.add.reduce(vectors)/vectors.shape[0]
    avg_v_hat = avg_v/np.linalg.norm(avg_v)
    # get unit vectors of all vectors
    v_hats = vectors/np.linalg.norm(vectors)

    angles = 0
    for i in range(vectors.shape[0]):
        v_hat = vectors[i]/np.linalg.norm(vectors[i])
        cos_theta = np.clip(np.dot(avg_v_hat, v_hat), -1.0, 1.0)
        angles = angles + cos_theta*cos_theta
    
    average_theta = angles/vectors.shape[0]
    S = (3*average_theta-1)*0.5
    if np.isnan(S):
        print('order parameter S is NaN')
        #print('nematics', vectors)
        #print('angles', angles)
    
    #print('S -', S )
    return S



#########################################################################################################

def apply_function_to_sliding_window(arr, window_comps, touch_matrix, funct_list, primary_nematic, secondary_nematic, StandardDeviation = True):
    #empirical storage arrays
    # functions that I want: 
    # 1. average number of nuclei per window
    # 2. average number of neighbours per window
    # 3. average primary and secondary nematic angles (order parameter S)
    """ Takes a list of functions and iteratively applies them to each sliding window. To do this, it takes as inputs the image array, 
     the touch matrix and the primary and secondary nematics to produce:
        1. average number of nuclei per window
        2. average number of neighbours per window
        3. average primary and secondary nematic angles
    Returns a numpy array for each function and a numpy arraafy with the averaged values of number of nuclei over each pixel.
    """
    store_arrays = [np.zeros_like(arr, dtype=np.float16) for i in range(len(funct_list))]

    n_arrays = len(funct_list)
    n_arrays_std = 0

    if count_number_of_contacts_per_window in funct_list:
        n_arrays_std += 1

    if get_average_angle_between_nematics_per_window in funct_list:
        n_arrays += 1
        n_arrays_std += 2
        #store_arrays.append(np.zeros_like(arr, dtype=np.float16))

    if order_parameter_s in funct_list:
        n_arrays += 1
        #store_arrays.append(np.zeros_like(arr, dtype=np.float16))

    if get_angles_between_director_and_nematics_per_window in funct_list:
        n_arrays += 1
        n_arrays_std += 2
        #store_arrays.append(np.zeros_like(arr, dtype=np.float16))

    print('arrays', n_arrays, n_arrays_std)
    
    store_arrays = [np.zeros_like(arr, dtype=np.float16) for i in range(n_arrays)]
    
    if StandardDeviation:
        store_arrays_std = [np.zeros_like(arr, dtype=np.float16) for i in range(n_arrays_std)]
    
    nuclei_ref = np.zeros_like(arr)
    bool_ref = np.zeros_like(arr)
    new_arr = np.copy(arr)

    max_num_ids = 0
    
    for w in window_comps:
        z_coords = w[0]
        y_coords = w[1]
        x_coords = w[2]
        window = new_arr[z_coords[0]:z_coords[1], y_coords[0]:y_coords[1], x_coords[0]:x_coords[1]]

        ids = np.unique(window).astype(int)[1:]
        #print('number of ids in  window', len(ids))
        #print(len(primary_nematic[0]))

        if len(ids) > max_num_ids:
            max_num_ids=len(ids)

        if len(ids) <= 3:
            continue

        else:
            #i_arr = np.zeros_like(ids)
            i_arrays = [np.zeros_like(ids) for i in range(len(funct_list)+1)]
            esempio = [0 for i in range(len(funct_list)+1)]
            # the first position is dedicated to the number of nuclei in the window
            num_nuclei = count_nuclei_per_window(ids)

            if get_average_angle_between_nematics_per_window in funct_list:
                primary_arrays, secondary_arrays = [],[]

            primary_nematics_in_window, secondary_nematics_in_window = [], []

            for j in range(len(ids)):
                for i in range(len(funct_list)):
                    funct = funct_list[i]
                    if funct is count_number_of_contacts_per_window:
                        i_arrays[i][j] = funct(ids[j], touch_matrix)
                        esempio[i] = 1
                    elif funct is get_average_angle_between_nematics_per_window:
                        (angles_prim, angles_sec) = funct(j, ids, touch_matrix, primary_nematic, secondary_nematic)
                        if len(angles_prim) >= 1:
                            primary_arrays.extend(angles_prim)
                            secondary_arrays.extend(angles_sec)
                    elif funct is order_parameter_s or get_angles_between_director_and_nematics_per_window:
                        #print(ids, ids-j)
                        pn = abs(primary_nematic[0][ids[j]-1]-primary_nematic[1][ids[j]-1])
                        sn = abs(secondary_nematic[0][ids[j]-1]-secondary_nematic[1][ids[j]-1])
                        #print('pn and sn', pn, sn)
                        if not np.all(pn== 0):
                            primary_nematics_in_window.append(pn)
                        if not np.all(sn == 0):
                            secondary_nematics_in_window.append(sn)
            
            primary_nematics_in_window = np.asarray(primary_nematics_in_window)
            secondary_nematics_in_window = np.asarray(secondary_nematics_in_window)
 
            if primary_nematics_in_window.shape[0] > 0:
                if get_angles_between_director_and_nematics_per_window in funct_list:
                    angles_between_director_and_primary_nematics = get_angles_between_director_and_nematics_per_window(np.asarray(primary_nematics_in_window))
                    angles_between_director_and_secondary_nematics = get_angles_between_director_and_nematics_per_window( np.asarray(secondary_nematics_in_window))
                if order_parameter_s in funct_list:
                    parameter_s_prim = order_parameter_s(np.asarray(primary_nematics_in_window))
                    parameter_s_sec = order_parameter_s(np.asarray(secondary_nematics_in_window))
                         
            
            for i in range(len(funct_list)):
                funct = funct_list[i]
                pos_sec = len(funct_list)-i
                if funct is count_number_of_contacts_per_window:
                    store_arrays[i][w[0][0]:w[0][1], w[1][0]:w[1][1], w[2][0]:w[2][1]] = store_arrays[i][w[0][0]:w[0][1], w[1][0]:w[1][1], w[2][0]:w[2][1]] + np.mean(i_arrays[i])
                if funct is get_average_angle_between_nematics_per_window:
                    store_arrays[i][w[0][0]:w[0][1], w[1][0]:w[1][1], w[2][0]:w[2][1]] = store_arrays[i][w[0][0]:w[0][1], w[1][0]:w[1][1], w[2][0]:w[2][1]] + np.mean(primary_arrays)
                    store_arrays[-pos_sec][w[0][0]:w[0][1], w[1][0]:w[1][1], w[2][0]:w[2][1]] = store_arrays[-pos_sec][w[0][0]:w[0][1], w[1][0]:w[1][1], w[2][0]:w[2][1]] + np.mean(secondary_arrays)
                if funct is order_parameter_s:
                    store_arrays[i][w[0][0]:w[0][1], w[1][0]:w[1][1], w[2][0]:w[2][1]] = store_arrays[i][w[0][0]:w[0][1], w[1][0]:w[1][1], w[2][0]:w[2][1]] + parameter_s_prim
                    store_arrays[-pos_sec][w[0][0]:w[0][1], w[1][0]:w[1][1], w[2][0]:w[2][1]] = store_arrays[-pos_sec][w[0][0]:w[0][1], w[1][0]:w[1][1], w[2][0]:w[2][1]] +parameter_s_sec
                if funct is get_angles_between_director_and_nematics_per_window:
                    store_arrays[i][w[0][0]:w[0][1], w[1][0]:w[1][1], w[2][0]:w[2][1]] = store_arrays[i][w[0][0]:w[0][1], w[1][0]:w[1][1], w[2][0]:w[2][1]] + np.median(angles_between_director_and_primary_nematics)
                    store_arrays[-pos_sec][w[0][0]:w[0][1], w[1][0]:w[1][1], w[2][0]:w[2][1]] = store_arrays[-pos_sec][w[0][0]:w[0][1], w[1][0]:w[1][1], w[2][0]:w[2][1]] + np.median(angles_between_director_and_secondary_nematics)


            if StandardDeviation:
                c = 0
                for i in range(len(funct_list)):
                    funct = funct_list[i]
                    if funct is count_number_of_contacts_per_window:
                        store_arrays_std[c][w[0][0]:w[0][1], w[1][0]:w[1][1], w[2][0]:w[2][1]] = store_arrays_std[c][w[0][0]:w[0][1], w[1][0]:w[1][1], w[2][0]:w[2][1]] + np.std(i_arrays[i])
                        c = c+1
                    if funct is get_average_angle_between_nematics_per_window:
                        store_arrays_std[c][w[0][0]:w[0][1], w[1][0]:w[1][1], w[2][0]:w[2][1]] = store_arrays_std[c][w[0][0]:w[0][1], w[1][0]:w[1][1], w[2][0]:w[2][1]] + np.std(primary_arrays)
                        store_arrays_std[c+1][w[0][0]:w[0][1], w[1][0]:w[1][1], w[2][0]:w[2][1]] = store_arrays_std[-pos_sec][w[0][0]:w[0][1], w[1][0]:w[1][1], w[2][0]:w[2][1]] + np.std(secondary_arrays)
                        c = c+2
                    if funct is get_angles_between_director_and_nematics_per_window:
                        store_arrays_std[c][w[0][0]:w[0][1], w[1][0]:w[1][1], w[2][0]:w[2][1]] = store_arrays_std[c][w[0][0]:w[0][1], w[1][0]:w[1][1], w[2][0]:w[2][1]] + np.std(angles_between_director_and_primary_nematics)
                        store_arrays_std[c+1][w[0][0]:w[0][1], w[1][0]:w[1][1], w[2][0]:w[2][1]] = store_arrays_std[c+1][w[0][0]:w[0][1], w[1][0]:w[1][1], w[2][0]:w[2][1]] + np.std(angles_between_director_and_secondary_nematics)
                        c = c+2


            nuclei_ref[w[0][0]:w[0][1], w[1][0]:w[1][1], w[2][0]:w[2][1]] = nuclei_ref[w[0][0]:w[0][1], w[1][0]:w[1][1], w[2][0]:w[2][1]] + num_nuclei
            bool_ref[w[0][0]:w[0][1], w[1][0]:w[1][1], w[2][0]:w[2][1]] = bool_ref[w[0][0]:w[0][1], w[1][0]:w[1][1], w[2][0]:w[2][1]] + 1
    
    
    smoothed_arrays = [nuclei_ref/bool_ref]
    for i in range(len(store_arrays)):
        smoothed_arrays.append(store_arrays[i]/bool_ref)
    
    
    if not StandardDeviation:
        return (smoothed_arrays, 0)
    elif StandardDeviation:
        smoothed_arrays_std = []
        for i in range(len(store_arrays_std)):
            smoothed_arrays_std.append(store_arrays_std[i]/bool_ref)

        print("Smoothed arrays", len(smoothed_arrays), "and std ", len(smoothed_arrays_std))
        return (smoothed_arrays, smoothed_arrays_std)

# ------------------------------------

def sliding_window_through_image(img, resolution, primary_nematics, secondary_nematics, size_window=10, StandardDeviation = True, verbose=True):
    """ Takes a labelled image with n labels and an integer number for expanding each labelled object.
    Returns an (n+1)*(n+1) matrix filled with 0s and each row and column correspond to one label. Where object i and object j
    touch, the position matrix[i][j] is equal to 1. Returns a list of number of neighbours per labelled object.
    """
     # define windows
    [z_res, y_res, x_res] = resolution
   
    window_size = define_window_size(z_res, y_res, x_res, size_micron=size_window)
    shift_step = define_shift_step(window_size)
  
    
    stopwatch('Define window')


    if verbose:
        print("Window size:", window_size)
        print("Shift step: ", shift_step)
        print('Image size: ', img.shape)
        print(test_number_of_steps_per_image(img.shape, window_size, shift_step))
    
    windows = sliding_window(img, window_size, shift_step)
    if verbose:
        print("How many windows? ", len(windows))
    stopwatch("Windows")

    #touch matrix
    touch_matrix, _ = get_touch_matrix(img, dilation=1)
    touch_matrix = touch_matrix.astype(int)
    
    
    total_ids = np.unique(img)[1:]
    if verbose:
        print('Number of primary nematics and ids --->', len(primary_nematics[0]), len(total_ids))
    
  
    all_smoothed_arrays = apply_function_to_sliding_window(img, windows, touch_matrix, 
                                    [count_number_of_contacts_per_window, order_parameter_s, get_angles_between_director_and_nematics_per_window], 
                                    primary_nematics, secondary_nematics, StandardDeviation=StandardDeviation)

    
    return  all_smoothed_arrays