"""
@authors:   Lucrezia Ferme @ Norden group @ IGC
            Allyson Quinn Ryan @ Modes group @ MPI-CBG and @ Haase group @ TU Dresden
@descript:  Functions for the extraction of several nuclear features to be used
            with regionprops.
"""

""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import math
import numpy as np
from numpy.linalg import norm
from glob import glob
from tifffile import imread
from scipy.ndimage import center_of_mass
from skimage.segmentation import find_boundaries
from skimage.measure import marching_cubes, mesh_surface_area, regionprops, label
from skimage.morphology import cube
from scipy import ndimage
from scipy.spatial.distance import cdist, squareform, pdist
import matplotlib.pyplot as plt
from sliding_windows_averaging import *
from preprocess import preprocessing, saving_tiff, clearing, get_spatial_calib_tiff



####################################################################################################################################
###                                                  REGIONPROPS FUNCTIONS
#####################################################################################################################################



def get_boundary(region):
    """Get coordinates of the object boundaries"""
    px_bnds = np.where(find_boundaries(region, connectivity = 3))
    bnds = np.asarray(px_bnds)
    return np.asarray(bnds).transpose()


#-------------------------------------------------------------------------------

def index_pair_from_condensed_form(array, condensed_idx):
    """ Gets the index of the two touching objects from a condensed array"""
    i = condensed_idx
    n = array.shape[0] - 1
    t_n = (n * (n+1)) / 2
    y = t_n - i - 1
    d = 1 + int((((8 * y + 1) ** 0.5) - 1) / 2)
    k = n - d
    k_star = 1 + i + k + ((d * (d + 1)) / 2) - t_n
    return int(k), int(k_star)


#-------------------------------------------------------------------------------

def primary_nematic(region):
    boundaries = get_boundary(region)
    distances = pdist(boundaries).astype(np.float16)
    idx = np.argmax(distances)
    idx_1, idx_2 = index_pair_from_condensed_form(boundaries, idx)
    coords11, coords12 = boundaries[idx_1, ...], boundaries[idx_2, ...]
    return coords11, coords12, boundaries


#-------------------------------------------------------------------------------

def nematics(region):
    def find_points_in_plane(p, v, p0):
        ''' equation of a plane perpendicular to vector (A, B, C) and passing through
         point (x0, y0, z0) is A(x-x0)+B(y-y0)+C(z-z0) = 0
         p are the points composint the boundaries, v the vector and p0 the point'''
        points = []
        for i in range(len(p)):
            #if np.rint(sum(v*(p[i]-p0))) == 0:
            if -10< np.rint(sum(v*(p[i]-p0))) < 10:  #21.05.22  changed from 5 to 10
                points.append(p[i])
        return np.array(points)

    def orthogonal(A, B):
        if 0 <= abs(np.dot(A, B)/(norm(A)*norm(B))) <= 0.01:
            return True
        else:
            return False

    def plot_nematics():
        """Plot to show the first and second nematic axes. The second nematic was calculated as the major axis
        on the perpendicular plane to the first nematic (I am going to ask Carl Modes if it makes sense)."""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        #plt.style.use("dark_background")
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(boundaries[::10, 2], boundaries[::10,1], boundaries[::10,0], color='blue', alpha=0.2) #'lightseagreen'
        ax.scatter(points[:,2], points[:,1], points[:,0], color='r', alpha=0.8)
        ax.plot([coords11[2], coords12[2]], [coords11[1], coords12[1]], [coords11[0], coords12[0]], label="first nematic", color='black')
        ax.plot([coords21[2], coords22[2]], [coords21[1], coords22[1]], [coords21[0], coords22[0]], label="second nematic", color='black')
        plt.show()


    coords11, coords12, boundaries = primary_nematic(region)
    first_nematic = coords12 - coords11
    m = (coords11+coords12)/2
    points = find_points_in_plane(boundaries, first_nematic, m)
    if len(points) > 1:
        distances_ = pdist(points)
        idx_ = np.argmax(distances_) #fix: what happens if the array distances_ is empty?
        idx_a, idx_b = index_pair_from_condensed_form(points, idx_)
        coords21, coords22 = points[idx_a, ...], points[idx_b, ...]
        #if not orthogonal(first_nematic, coords22-coords21):
            #print("The second nematic is not exactly perpendicular")
        plotting = False
        if plotting:
            plot_nematics()
        return (coords11, coords12), (coords21, coords22)
    elif len(points) <= 1:
        return (coords11, coords12), (np.asarray([None, None, None]), np.asarray([None, None, None]))


#-------------------------------------------------------------------------------

def sphericity_by_volume(region):
    def sphere_volume(r):
        V = (4/3)*np.pi*(r**3)
        return V

    bnds = np.where(find_boundaries(region, connectivity = 3))
    set_bnds = np.asarray(list(set([(bnds[0][j], bnds[1][j], bnds[2][j]) for j in range(bnds[0].shape[0])])))
    dists_bnds_bnds = pdist(set_bnds) #retry with set_bnds just to be sure

    d = np.amax((dists_bnds_bnds))
    max_radius = d * 0.5
    ref_vol = sphere_volume(max_radius)
    obj_vol = np.sum(region)
    return obj_vol/ref_vol


#-------------------------------------------------------------------------------

def sphericity_by_surface(region):
    """Computes the ratio between the surface area of the given object and the
    the surface area of the sphere having the same volume of the object
    """

    verts, faces, normals, values = marching_cubes(region, 0)
    obj_surface_area = mesh_surface_area(verts, faces)

    obj_vol = np.sum(region)
    sphere_surface_area = np.float_power(np.pi, 1/3)*np.float_power(6*obj_vol, 2/3)


    return sphere_surface_area/obj_surface_area


#-------------------------------------------------------------------------------

def ellipsoid_axis_lengths(central_moments):
    """Computes ellipsoid major, intermediate and minor axis length.
    Parameters
    ----------
    central_moments : ndarray
        Array of central moments as given by ``moments_central`` with order 2.
    Returns
    -------
    axis_lengths: tuple of float
        The ellipsoid axis lengths in descending order. (not semiaxis)
    Code by Greg Lee
    https://forum.image.sc/t/scikit-image-regionprops-minor-axis-length-in-3d-gives-first-minor-radius-regardless-of-whether-it-is-actually-the-shortest/59273
    """
    m0 = central_moments[0, 0, 0]
    sxx = central_moments[2, 0, 0] / m0
    syy = central_moments[0, 2, 0] / m0
    szz = central_moments[0, 0, 2] / m0
    sxy = central_moments[1, 1, 0] / m0
    sxz = central_moments[1, 0, 1] / m0
    syz = central_moments[0, 1, 1] / m0
    S = np.asarray([[sxx, sxy, sxz], [sxy, syy, syz], [sxz, syz, szz]])
    # determine eigenvalues in descending order
    eigvals = np.sort(np.linalg.eigvalsh(S))[::-1]
    return tuple([math.sqrt(20.0 * e) for e in eigvals])

#-------------------------------------------------------------------------------

def euclidean_dist(P1, P2):
    axes = P2-P1
    axes2 = np.power(axes, 2)
    axes2sum = np.sum(axes2, axis=0)
    return np.power(axes2sum, 0.5)

#-------------------------------------------------------------------------------

def get_touch_matrix(img, dilation):
    """ Takes a labelled image with n labels and an integer number for expanding each labelled object.
    Returns an (n+1)*(n+1) matrix filled with 0s and each row and column correspond to one label. Where object i and object j
    touch, the position matrix[i][j] is equal to 1. Returns a list of number of neighbours per labelled object.
    See pyclesperanto prototype"""

    assert isinstance(dilation, int), "The dilation value should be an integer"
    # extend labels by a dilation factor
    extended_labels = cle.extend_labels_with_maximum_radius(img, radius = dilation)

    # get matrix of touching objects
    touch_matrix = cle.generate_touch_matrix(extended_labels)
    cle.set_column(touch_matrix, 0, 0)
    neighbors = cle.count_touching_neighbors(touch_matrix)[0]


    return touch_matrix, cle.pull(neighbors)

#-------------------------------------------------------------------------------

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

#-------------------------------------------------------------------------------

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2':
            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    Definition taken from
    https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python/13849249#13849249
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))



####################################################################################################################################
###                                                  FEATURES EXTRACTION FUNCTION
#####################################################################################################################################


def stopwatch(text : str = None):
    if not stopwatch.verbose:
        return
    import time

    if text is not None:
        print("-------------------------> " + text + " took " + str(int((time.time() - stopwatch.timestamp) * 1000)) + "ms")

    stopwatch.timestamp = time.time()

stopwatch.timestamp = 0
stopwatch.verbose = True

# --------------- --------------------------------------------------------- ------------------------------

def run(srcDir : str, dstDir : str, file : str,  dataDir: str , nemDir :str , neighDir : str, dilations = [0],
        save_images = True, verbose=True,
        singleNucleusFeatures = True,
        NeighbourStatistics = True,
        LocalAveraging = True,
        isotropicdata = False,
       ):
    """
    """

    if verbose:
        if singleNucleusFeatures:
            print("EXTRACTION OF SINGLE NUCLEAR FEATURES")
        if NeighbourStatistics:
            print("PERFORM STATISTICS OF NEIGHBOURS")
        if LocalAveraging:
            print("PERFORM SLIDING WINDOW AVERAGING")
        print("\n")


    #Generate output directories where results are saved
    print("Available OpenCL devices:" + str(cle.available_device_names()))

    isotropicDir = dstDir+'/isotropic_masks'

    # --------------------------------------------------------------------------
    # READ IMAGES:
    # --> mask is the original anisotropic image
    # --> rescaled_mask is the rescaled image (isotropic voxels)

    mask = np.array(imread(file), dtype='int')
    filename = os.path.basename(file)
    stage = int(filename[filename.find("hpf")-2:filename.find("hpf")])
    #stage = int(filename[filename.find("ss")-1:filename.find("ss")])
    stopwatch('A. Read image')

    resolution = get_spatial_calib_tiff(file)
    print('---> VOXEL SIZE', resolution)
    if isotropicdata:
        rescaled_mask = np.array(imread(isotropicDir+'/'+filename), dtype='int')
        rescaled_resolution = get_spatial_calib_tiff(isotropicDir+'/'+filename)
    else:
        rescaled_mask, rescaled_resolution = preprocessing(mask,
                                                        img_path=file, resolution=resolution)
    print('---> RESCALED VOXEL SIZE', rescaled_resolution)
    stopwatch('B. Preprocessing')

    # --------------------------------------------------------------------------
    # CLEAR IMAGES AND RELABEL THEM SEQUENTIALLY
    #relabel_mask = relabel_sequential(rescaled_mask)[0]
    clean_mask = clearing(rescaled_mask, min_size=200) #clear nuclei touching the image, no change in nuclear label
    relabelled_mask = relabel_sequential(clean_mask)[0] # relabel of the cleaned image
    stopwatch('C. Clearing')
    saving_tiff(np.array(relabelled_mask.astype(np.float32)), filename, outDir=isotropicDir, resolution=rescaled_resolution, verbose=True)
    

    original_ids = tuple(np.unique(clean_mask))[1:] # original labels
    ids = tuple(np.unique(relabelled_mask))[1:] # sequential labels after clearing
    if verbose:
        print("{} labels were found".format(len(ids)))
        print("Original labels", len(original_ids))


    if len(ids) == 0:
        return

   # temp_dict = {'original_label': original_ids}
   # temp_df = pd.DataFrame(temp_dict)
   # temp_df.to_csv(dataDir+"/"+filename[:-4]+".csv")




    # --------------------------------------------------------------------------
    # INITIALIZE DICTIONARIES AND ARRAYS

    if singleNucleusFeatures:
        mask_dict = {'label':[], 'original_label':original_ids,
                'hpf':[], 'volume_um':[], 'volume_um_aniso':[],
                #'sphericity_by_volume': [],'sphericity_by_surface': [],
                #'sphericity_by_volume_aniso': [],'sphericity_by_surface_aniso': [],
                'long_axis_um':[], 'short_axis_a_um':[], 'short_axis_b_um':[],
                'primary_nematic_length_um':[], 'secondary_nematic_length_um':[],
                'centroid_z':[], 'centroid_y':[], 'centroid_x':[]}
        primary_nematic_1, primary_nematic_2 = np.zeros((len(ids), 3)), np.zeros((len(ids), 3))
        secondary_nematic_1, secondary_nematic_2 = np.zeros((len(ids), 3)), np.zeros((len(ids), 3))
        stopwatch('Initializing dictionaries and arrays')


        # --------------------------------------------------------------------------
        # REGIONPROPS

        stats = regionprops(relabelled_mask, relabelled_mask, extra_properties = [sphericity_by_volume, sphericity_by_surface])
        stopwatch('Regionprops')

        # --------------------------------------------------------------------------
        # EXTRACT FEATURES

        for i in range(len(ids)):
            #print("---> ", i, " LABEL: ", ids[i])
            obj = relabelled_mask == ids[i]
            _, num = label(obj, return_num=True, connectivity=3)

            mask_dict['label'].append(stats[i].label)
            mask_dict['hpf'].append(stage)

            if np.sum(obj) > 0 and num == 1:
                bbox = stats[i].bbox
                z_min, y_min, x_min, z_max, y_max, x_max = bbox
                roi = np.copy(relabelled_mask[z_min:z_max, y_min:y_max, x_min:x_max])
                roi[roi != ids[i]] = 0

                anisotropic_obj = mask == original_ids[i]

                ##
                axes = ellipsoid_axis_lengths(stats[i]['moments_central'])
                mask_dict['volume_um'].append(stats[i].area * rescaled_resolution[0] * rescaled_resolution[1] * rescaled_resolution[2])
                mask_dict['volume_um_aniso'].append(np.sum(anisotropic_obj) * resolution[0] * resolution[1] * resolution[2])
               # mask_dict['sphericity_by_volume'].append(stats[i].sphericity_by_volume)
               # mask_dict['sphericity_by_surface'].append(stats[i].sphericity_by_surface)
                mask_dict['long_axis_um'].append(axes[0]*resolution[0])
                mask_dict['short_axis_a_um'].append(axes[1]*resolution[0])
                mask_dict['short_axis_b_um'].append(axes[2]*resolution[0])
                centroids = stats[i].centroid
                mask_dict['centroid_z'].append(centroids[0])
                mask_dict['centroid_y'].append(centroids[1])
                mask_dict['centroid_x'].append(centroids[2])
                #stopwatch('G. Fill in dictionary - before nematic')

                (coords11, coords12), (coords21, coords22) = nematics(roi)
                mask_dict["primary_nematic_length_um"].append(euclidean_dist(coords11, coords12)*resolution[0])

                primary_nematic_1[i,...] = (z_min, y_min, x_min) + coords11
                primary_nematic_2[i,...] = (z_min, y_min, x_min) + coords12 #zyx
                if None in coords21:
                    secondary_nematic_1[i, ...], secondary_nematic_2[i, ...] = coords21, coords22
                    mask_dict["secondary_nematic_length_um"].append(0)
                else:
                    secondary_nematic_1[i, ...] = (z_min, y_min, x_min) + coords21
                    secondary_nematic_2[i, ...] = (z_min, y_min, x_min) + coords22
                    mask_dict["secondary_nematic_length_um"].append(euclidean_dist(coords21, coords22)*resolution[0])


            else:
                mask_dict['volume_um'].append(0)
               # mask_dict['sphericity_by_volume'].append(0)
               # mask_dict['sphericity_by_surface'].append(0)
                mask_dict['volume_um_aniso'].append(0)
                mask_dict['long_axis_um'].append(0)
                mask_dict['short_axis_a_um'].append(0)
                mask_dict['short_axis_b_um'].append(0)
                mask_dict["primary_nematic_length_um"].append(0)
                mask_dict["secondary_nematic_length_um"].append(0)
                centroids=stats[i].centroid
                mask_dict['centroid_z'].append(centroids[0])
                mask_dict['centroid_y'].append(centroids[1])
                mask_dict['centroid_x'].append(centroids[2])
                primary_nematic_1[i,...], primary_nematic_2[i,...] = (0,0,0), (0,0,0) #zyx
                secondary_nematic_1[i, ...], secondary_nematic_2[i, ...] = (0,0,0), (0,0,0)

        df = pd.DataFrame(mask_dict)
        stopwatch('End of ids looping')

        # --------------------------------------------------------------------------
        # SAVE PRIMARY AND SECONDARY NEMATIC AXES COORDS

        primary = [primary_nematic_1, primary_nematic_2]
        secondary = [secondary_nematic_1, secondary_nematic_2]
        np.save(os.path.join(nemDir, filename[:-4]+'_primary-nematics.npy'), primary)
        np.save(os.path.join(nemDir, filename[:-4]+'_secondary-nematics.npy'), secondary)



    # --------------------------------------------------------------------------
    # GET NUMBER OF NEIGHBOURS PER DILATION

    #relabel_mask_gpu = cle.push_zyx(relabel_mask)
    clean_mask_gpu = cle.push_zyx(relabelled_mask)

    if singleNucleusFeatures:
        for d in dilations:
            _, number_of_neighbors = get_touch_matrix(clean_mask_gpu, d)
            correct_number_of_neighbors = [number_of_neighbors[i] for i in ids]
            df['number_of_neighbors_dilation_'+str(d)] = correct_number_of_neighbors
        stopwatch('Count neighbors')

        # --------------------------------------------------------------------------
        # SAVING DATAFRAME

        df.to_csv(dataDir+"/"+filename[:-4]+".csv")
        pointlists = cle.pull(cle.label_centroids_to_pointlist(clean_mask_gpu))
        np.save(os.path.join(dataDir, filename[:-4]+'_pointlist.npy'), pointlists)
        stopwatch("Saving nucleardata")

    # -------------------------------------------------------------------------
    # STATISTICS OF NEIGHBOURS

    if NeighbourStatistics:
        #distance_mesh = cle.draw_distance_mesh_between_touching_labels(clean_mask_gpu)
        stats = pd.DataFrame(cle.statistics_of_labelled_neighbors(clean_mask_gpu))
        if stats is not None:
            stats.to_csv(os.path.join(neighDir,filename[:-4]+"_stats_of_neighbours.csv"))


    # --------------------------------------------------------------------------
    # WINDOW AVERAGING

    if LocalAveraging and singleNucleusFeatures:
       # size_window_micron = 10
        #[smoothed_arr_num_nuclei, smoothed_arr_neigh, smoothed_arr_param_s_prim, smoothed_arr_param_s_sec] = sliding_window_through_image(relabelled_mask, 
           #                                                                                                     primary_nematics = primary, secondary_nematics = secondary, 
             #                                                                                                   resolution = resolution, size_window= size_window_micron)
       # stopwatch('Sliding window through full image: completed')
       # imwrite(os.path.join(neighDir, filename[:-4]+'_count_of_neighbours_averaging.tif'), smoothed_arr_neigh)
       # imwrite(os.path.join(neighDir, filename[:-4]+'_count_of_nuclei_per_window.tif'), smoothed_arr_num_nuclei)
       # imwrite(os.path.join(nemDir, filename[:-4]+'_order_parameter_s_primary_nematics.tif'), smoothed_arr_param_s_prim)
       # imwrite(os.path.join(nemDir, filename[:-4]+'_order_parameter_s_secondary_nematics.tif'), smoothed_arr_param_s_sec)
                                                                                                                

        size_window_micron = 10
        (smoothed_arrays, smoothed_arrays_std) = sliding_window_through_image(relabelled_mask, primary_nematics = primary, secondary_nematics = secondary, 
                                                                                resolution = resolution, size_window= size_window_micron, StandardDeviation=True)
        stopwatch('B.Sliding window 10 um through full image: completed')

        [smoothed_arr_num_nuclei, smoothed_arr_neigh, smoothed_arr_param_s_prim, smoothed_arr_director_prim, smoothed_arr_param_s_sec,smoothed_arr_director_sec] = smoothed_arrays
        [smoothed_arr_neigh_std,  smoothed_arr_director_prim_std,smoothed_arr_director_sec_std] = smoothed_arrays_std

        # averages
        imwrite(os.path.join(neighDir, filename[:-4]+'_count_of_neighbours_per_window_10.tif'), smoothed_arr_neigh)
        imwrite(os.path.join(neighDir, filename[:-4]+'_count_of_nuclei_per_window_10.tif'), smoothed_arr_num_nuclei)
        imwrite(os.path.join(nemDir, filename[:-4]+'_order_parameter_s_primary_nematics_window_10.tif'), smoothed_arr_param_s_prim)
        imwrite(os.path.join(nemDir, filename[:-4]+'_order_parameter_s_secondary_nematics_window_10.tif'), smoothed_arr_param_s_sec)
        imwrite(os.path.join(nemDir, filename[:-4]+'_angle_director_primary_nematics_window_10.tif'), smoothed_arr_director_prim)
        imwrite(os.path.join(nemDir, filename[:-4]+'_angle_director_secondary_nematics_window_10.tif'), smoothed_arr_director_sec)
        # std 
        imwrite(os.path.join(neighDir, filename[:-4]+'_std_count_of_nuclei_per_window_10.tif'), smoothed_arr_neigh_std)
        imwrite(os.path.join(nemDir, filename[:-4]+'_std_angle_director_primary_nematics_window_10.tif'), smoothed_arr_director_prim_std)
        imwrite(os.path.join(nemDir, filename[:-4]+'_std_angle_director_secondary_nematics_window_10.tif'), smoothed_arr_director_sec_std)

        stopwatch('B.Saving images')
        
    

    return
