"""
@authors:   Lucrezia Ferme @ Norden group @ IGC
@descript:  Functions to process StarDist-3D predictions before feature extraction
"""

import os
import copy
import numpy as np
from PIL import Image
from PIL.TiffTags import TAGS
from skimage.morphology import remove_small_objects, skeletonize
from skimage.segmentation import find_boundaries, clear_border
import pyclesperanto_prototype as cle
from tifffile import imsave, imwrite

#-----------------------------------------------------------------------------------------------------------------------

def get_spatial_calib_tiff(image):
    """ Gets the path/ title of a tiff file and returns the zyx pixel sizes in microns"""
    with Image.open(image) as img:
        meta_dict = {TAGS[key] : img.tag[key] for key in img.tag.keys()}

    # unit: micron
    z = [w[-4:] for w in meta_dict['ImageDescription'][0].split("\n") if w.startswith('spacing')]
    x = 1/ (meta_dict["XResolution"][0][0]/meta_dict["XResolution"][0][1])
    y = 1/ (meta_dict["YResolution"][0][0]/meta_dict["YResolution"][0][1])
   

    return float(z[0]), float("%.4f" % y), float("%.4f" % x)


#-----------------------------------------------------------------------------------------------------------------------

""
def clearing(image, min_size=64):
    """Returns a label image image that has been cleared of objects touching the image borders
    and objects that are smaller than a given minimal size """
    img = image.copy()
    img = clear_border(img, bgval = 0)
    img = remove_small_objects(img, min_size = min_size, connectivity = 2)
    #ids = tuple(np.unique(img))
    return img

#-----------------------------------------------------------------------------------------------------------------------

""
def saving_tiff(img, filename, outDir, resolution, verbose=False):
    imsave(os.path.join(outDir, filename), img, imagej=True, resolution=(1/resolution[2], 1/resolution[1]),
            metadata={'spacing': resolution[0],'axes': 'ZYX','unit': 'um'})
    if verbose:
        print("Image {} saved in directory: {}".format(filename[:-4], outDir))
    return

#-----------------------------------------------------------------------------------------------------------------------

""
def rescaling(img_gpu, voxel_size=None, z=1, y=1, x=1):
    """Rescales the label image to make it (almost) isotropic
    Returns a processed label image and rescaled calibration
    Parameters
    ----------
    mask: """

    if voxel_size == None:
        factor_xy = x/z
        factor_z = 1
    else:
        factor_xy = x/voxel_size
        factor_z = z/voxel_size

    # rescale image to have a resolution in
    resampled = cle.create([int(img_gpu.shape[0]*factor_z), int(img_gpu.shape[1] * factor_xy), int(img_gpu.shape[2] * factor_xy)])
    cle.scale(img_gpu, resampled, factor_x= factor_xy, factor_y= factor_xy, factor_z=factor_z, centered=False)

    rescaled_resolution = (float("%.4f" % ((img_gpu.shape[0]*z)/resampled.shape[0])),
                            float("%.4f" % ((img_gpu.shape[1]*y)/resampled.shape[1])),
                            float("%.4f" % ((img_gpu.shape[2]*x)/resampled.shape[2])))

    return resampled, rescaled_resolution


######################################################################################################################
######################################################################################################################



def preprocessing(mask, img_path=None,  dstDir=None, resolution = [], min_size=0, rescale= True,
                    saveimg = False, verbose=False):

    # Get pixel resolution
    if len(resolution) <= 0:
        z_res, y_res, x_res = get_spatial_calib_tiff(img_path)
    elif len(resolution) > 0:
        [z_res, y_res, x_res] = resolution
        print("This is the resolution", z_res, y_res)


    # if a min_size is given, remove objects smaller than the threshold
    if min_size > 0:
        mask = remove_small_objects(mask, min_size= min_size, in_place = True, connectivity = 2)

    # push label image on gpu and process it
    if rescale:
        mask_gpu = cle.push_zyx(mask)
        #mask_gpu = cle.close_index_gaps_in_label_map(mask_gpu)
        mask_gpu, rescaled_resolution = rescaling(mask_gpu, z=z_res, y=y_res, x=x_res)
        new_mask = cle.pull(mask_gpu).astype(int)
        if saveimg:
            #try:
            filename = os.path.basename(img_path)
            m = np.array(new_mask.astype(np.float32))
            imwrite(os.path.join(dstDir, filename), m, imagej=True,
                resolution=(1/rescaled_resolution[2], 1/rescaled_resolution[1]),
                metadata={'spacing': rescaled_resolution[0], 'unit': 'um', 'axes': 'ZYX'})
            #except TypeError:
            #    print("You did not provide a correct path to a directory")
        return new_mask, rescaled_resolution
    else:
        return mask, (z_res, y_res, x_res)
        
#-----------------------------------------------------------------------------------------------------------------------