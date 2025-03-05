import numpy as np
import napari
import pandas as pd
import seaborn as sns
from glob import glob
from skimage.measure import regionprops, label
from scipy import ndimage as ndi
import pyclesperanto_prototype as cle
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes
from tifffile import imread, imsave, imwrite
from skimage.morphology import closing, cube, erosion
from skimage.measure import regionprops, label
import warnings
from napariviewer import *
from preprocess import get_spatial_calib_tiff



####################################################################################################################################
###                                                  NUCLEAR PACKING FRACTION
#####################################################################################################################################

def packing_volume(roi):
    roi_volume = roi.shape[0]*roi.shape[1]*roi.shape[2]
    volumes = np.sum(roi > 0)
    return volumes/roi_volume

#-----------------------------------------------------------------------------------------------------------------------
def zetaslice(img):
    z_min = int(input("Provide number of slice (start-min): "))
    z_max = int(input("Provide number of slice (end-max): "))
    return img[z_min:z_max, :, :]

#-----------------------------------------------------------------------------------------------------------------------

def measure_volume_fraction_from_masking(mask, radius=20):
    mgpu = cle.push_zyx(mask)
    ext_mgpu = cle.extend_labels_with_maximum_radius(mgpu, radius = radius)
    ext_img = closing(cle.pull(ext_mgpu), cube(100))
    ext_img = erosion(binary_fill_holes(ext_img), cube(70)).astype(int) #it was 100 before
    components, num = label(ext_img, return_num=True, connectivity=3)

    propsa = regionprops(components)
    labels = [label.label for label in propsa]
    volumes = [label.area for label in propsa]

    max_id = np.argmax(volumes)
    max_mask = np.where(components == labels[max_id], 1, 0)
    max_vol = volumes[max_id]

    return max_mask, max_vol


####################################################################################################################################
###                                                  RADIAL DISTRIBUTION FUNCTION
#####################################################################################################################################

def select_nuclei_in_roi(img, labels):
    from skimage.measure import regionprops
    stats = regionprops(img)
    new_img = np.zeros_like(img)
    for i in range(len(stats)):
        if stats[i].label in labels:
            bbox = stats[i].bbox
            z_min, y_min, x_min, z_max, y_max, x_max = bbox
            roi = img[z_min:z_max, y_min:y_max, x_min:x_max]
            mask = (roi == stats[i].label)
            new_img[z_min:z_max,y_min:y_max, x_min:x_max][mask] = roi[mask]
    return new_img

#-----------------------------------------------------------------------------------------------------------------------

def radial_distribution_function_3D(x, y, z, volMask, rMax, dr, interior_indices):
    """Computes the three-dimensional radial distribution function for a number of 
    objects (in this case nuclei) within a given volume. Takes the arrays of x, y and
    z positions of centroids, the volume of the mask (volMask), the outer diameter 
    of the largest shell (rMax), the increment for increasing the radius of the 
    spherical shell (dr) and the list of the objects to analyse (interior_indices).
    Returns a tuple: (g, radii, interior_indices)
        g(r)            a numpy array containing the correlation function g(r)
        radii           a numpy array containing the radii of the
                        spherical shells used to compute g(r)
    """
    from numpy import zeros, sqrt, where, pi, mean, arange, histogram

    num_interior_particles = len(interior_indices)

    #print('ALL PARTICLES', len(x))
    #print("INTERIOR", num_interior_particles)
    edges = arange(0., rMax + 1.1 * dr, dr)
    #print('edges', edges)
    
    num_increments = len(edges) - 1

    if num_interior_particles < 1:
        warnings.warn("No particles found for which a sphere of radius rMax\
                will lie entirely within a cube of side length S.  Decrease rMax\
                or increase the size of the cube.")
        empty_arr = np.empty((1,num_increments))
        empty_arr.fill(np.nan)
        
        return  (empty_arr[0], empty_arr[0],  interior_indices)

    g = np.empty([num_interior_particles, num_increments])
    g[:] = np.nan
    print('G', len(g), len(g[0]))
    radii = zeros(num_increments)
    numberDensity = (num_interior_particles / volMask )#*0.001 #(len(x) / volMask )*0.001
    print('numDensity', numberDensity, len(x), volMask)

    ### Compute pairwise correlation for each interior particle
    for p in range(num_interior_particles):
        index = interior_indices[p]-1
        d = sqrt((x[index] - x)**2 + (y[index] - y)**2 + (z[index] - z)**2)
        #print('d', d)
        d[index] = 2 * rMax
        

        (result, bins) = histogram(d, bins=edges)
        #plt.hist(d, bins=edges)
        #plt.show()
        #print('----', result, bins)
        g[p,:] = result / numberDensity

    ### Average g(r) for all interior particles and compute radii
    g_average = zeros(num_increments)
    for i in range(num_increments):
        radii[i] = (edges[i] + edges[i+1]) / 2.
        rOuter = edges[i + 1]
        rInner = edges[i]
        g_average[i] = np.nanmean(g[:, i]) / (4.0 / 3.0 * pi * (rOuter**3 - rInner**3))

    return (g_average, radii)

####################################################################################################################################
###                                                  ORDER PARAMETER S
#####################################################################################################################################

def order_parameter_s(directors):
    avg_director = np.add.reduce(directors)/directors.shape[0]
   # print('average_direction', avg_director)
    #avg_director_hat = avg_director/np.linalg.norm(avg_director)

    angles = np.zeros(directors.shape[0])
    director = directors[~np.isnan(directors)]
    for i in range(directors.shape[0]):
        nematic_director = directors[i]
        #print(nematic_director)
        theta = angle_between(nematic_director, avg_director)
        angles[i] = np.cos(theta)*np.cos(theta) #theta

    #print('angles', angles)
    #average_theta = np.nanmean(angles)
    S = (3*np.nanmean(angles)-1)/2
    #print("AVERAGE", np.degrees(average_theta), average_theta)
    
   # S = (3*np.cos(average_theta)*np.cos(average_theta)-1)/2
  #  print("S", S)
    return S