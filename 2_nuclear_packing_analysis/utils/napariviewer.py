import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import tifffile
import napari
import glob
from skimage import transform, measure
from skimage.measure import regionprops
from scipy import ndimage
import random
import warnings
from preprocess import get_spatial_calib_tiff

########################################################################################################################
# #######################################################################################################################




#-----------------------------------------------------------------------------------------------------------------------


def get_aspect_ratio(calib_orig):
    """Aspect ratio z/x for an image with this original calib which was downsized in x,y afterwards.
    Needed for napari.
    calib_orig: [x,y,z]"""
    aspect=calib_orig[2]/(calib_orig[0])
    return aspect

#-----------------------------------------------------------------------------------------------------------------------


def extract_and_prep_roi(shapes_layer, max_r, max_c):
    """ Processes data obtained from a napari 'shapes layer'.
    Extracts the rectangle roi coordinates of the shapes_layer. Upscales it to the original
    image dimensions and can take care of anisotropy. Clips to valid range.
    If no roi exists in the shape layer the full image size is returned as roi: 0,max_r,0,max_c
    params:
    shapes_layer: napari shapes layer
    scale_r: scale factor to convert row coordinates from roi to original image dimensions (from downscaling and/or
            anisotropy)
    scale_c: similar but for column coordinates
    max_r: maximum possible value for rows (from image shape)
    max_c: similar, for columns
    returns: [rmin,rmax,cmin,cmax]
    """
    shapes=shapes_layer.data
    print("rectangle shape", shapes)

    if len(shapes)>=1:
        roi=shapes[0]
        # make sure it's a rectangle & convert to int
        rmin=int(min(roi[:,0]))
        rmax=int(np.round(max(roi[:,0])))
        cmin=int(min(roi[:,1]))
        cmax=int(np.round(max(roi[:,1])))
        print("rectangle ", rmin, rmax, cmin, cmax)



        # clip to image dims
        rmin=max(0,rmin)
        rmax=min(max_r,rmax)
        cmin=max(0,cmin)
        cmax=min(max_c,cmax)
        print("Roi coordinates (px): top-left:(",rmin,",",cmin,"),  bottom-right: (",rmax,",",cmax,
              "). Limits from image dim: :(",max_r,",",max_c,")" )
        print("xy dimensions ", rmax-rmin, cmax-cmin)

    else:
        rmin=0
        rmax=max_r
        cmin=0
        cmax=max_c
        print("No cropping Roi selected. Using full range: top-left:(",rmin,",",cmin,"),  bottom-right: (",rmax,",",cmax, ")")

    return [rmin,rmax,cmin,cmax]


#-----------------------------------------------------------------------------------------------------------------------


def select_region(stack):
    shapes_layer = display_for_crop(stack, mode="XY")
    [ymin, ymax, xmin, xmax]= extract_and_prep_roi(shapes_layer, stack.shape[1], stack.shape[2])
    [zmin, zmax] = 0, stack.shape[0]
    coords = [zmin, zmax, ymin, ymax, xmin, xmax]
    return stack[coords[0]:coords[1],coords[2]:coords[3],coords[4]:coords[5]]

#-----------------------------------------------------------------------------------------------------------------------


def draw_line(img):
    img_mpi = img.max(axis=0)
    viewer = napari.view_image(img_mpi, name='roi')
    line_layer = viewer.add_shapes(name='line', face_color='red')
    line_layer.mode= "LINE"
    napari.run()
    line_array = viewer.layers['line'].data
  
    return line_array

#-----------------------------------------------------------------------------------------------------------------------


def display_for_crop(img, mode='XY', shape_type='rectangle', shape_size=None, copy=False):

    if mode == 'XY':
        dispimg = img
    else:
        dispimg=np.swapaxes(img,0,1)


    if shape_size is None:
        nslices, h, w = dispimg.shape
    else:
        try:
            check = isinstance(shape_size, list)
        except Warning: print('The shape size should be an object list')
        nslices, h, w = shape_size[0], shape_size[1], shape_size[2]

    offset=10
    roi_init= np.array([[offset,offset],[h-offset, offset], [h-offset, w-offset],[offset,w-offset]])
    shapes = [roi_init]


    # show image
    viewer = napari.Viewer()
    viewer.add_image(dispimg, name='image', colormap='inferno')

    # show rectangle
    shapes_layer = viewer.add_shapes(shapes, name='Roi'+mode, shape_type=shape_type, edge_width=2, # orange
                              edge_color='coral', face_color='#ffb17d48')#,opacity=0.7)
    shapes_layer.mode="SELECT"
    napari.run()

    return shapes_layer

#-----------------------------------------------------------------------------------------------------------------------


def get_angle_of_rotation(img):
    img_mpi = img.max(axis=0)
    viewer = napari.view_image(img_mpi, name='roi')
    viewer.add_shapes(name='line', face_color='red')
    napari.run()
    line_array = viewer.layers['line'].data
   
    y1, x1, y2, x2 = line_array[0][0][0], line_array[0][0][1], line_array[0][1][0], line_array[0][1][1]
    angle = -((180.0/np.pi)*np.arctan2(y1-y2, x2-x1)-90)
    return angle


#-----------------------------------------------------------------------------------------------------------------------


def rotate_image(img,angle, reshape=True):
    """Rotates image by a given angle"""
    return ndimage.rotate(img, angle, axes=(1, 2), reshape=reshape, order=0)


#-----------------------------------------------------------------------------------------------------------------------


def define_roi(img, rotation = False, smaller = True, verbose = False):
 
    if rotation:
        angle = get_angle_of_rotation(img)
        if verbose:
                print("The angle of rotation based on xy axes is ", angle)

        # rotate image
        rotated = rotate_image(img, angle)
    else:
        rotated = img.copy()

    # display rotated image and define coordinates for cropping in xy
    coords_xy = display_for_crop(rotated, "XY")
    [ymin, ymax, xmin, xmax] = extract_and_prep_roi(coords_xy,rotated.shape[1],rotated.shape[2])

    # display the cropped roi and define coordinates to crop in z
    coords_z = display_for_crop(rotated[:, ymin:ymax, xmin:xmax], mode="Z")
    [zmin, zmax, _, _] = extract_and_prep_roi(coords_z,rotated.shape[0],rotated.shape[2])
    cropped = rotated[zmin:zmax, ymin:ymax, xmin:xmax]
    coords = [zmin, zmax, ymin, ymax, xmin, xmax]

    if verbose:
        print("Cropping done. The new shape of the roi is (ZYX) ", cropped.shape)


    return cropped, coords

#-----------------------------------------------------------------------------------------------------------------------


def define_smaller_roi(roi, coords, verbose=True):
    zmin, zmax, ymin, ymax, xmin, xmax = coords
    a, b, c = int((xmax-xmin)/8), int((ymax-ymin)/8), int((zmax-zmin)/10)
    print('zmax', zmax)
    margin_x = random.randrange(5, a)
    margin_y = random.randrange(5, b)
    margin_z = random.randrange(1, c)

    print(zmin+margin_z, zmax-margin_z)
    print(ymin+margin_y, ymax-margin_y)
    print(xmin+margin_x, xmax-margin_x)

    smaller_roi = roi[margin_z:roi.shape[0]-margin_z, margin_y:roi.shape[1]-margin_y, margin_x:roi.shape[2]-margin_x]
    print("smaller", smaller_roi.shape)
    print("roi", roi.shape)

    return smaller_roi

#-----------------------------------------------------------------------------------------------------------------------

def zetaslice(img):
    z_min = int(input("Provide number of slice (start-min): "))
    z_max = int(input("Provide number of slice (end-max): "))
    return img[z_min:z_max, :, :]

#-----------------------------------------------------------------------------------------------------------------------

def select_polygon(img):

    img = np.array(img, dtype='int')

    max_r, max_c = img.shape[1], img.shape[2]

    viewer = napari.view_labels(img, name='labeled image')
    selection_layer = viewer.add_shapes([[[0, 0], [128,0], [128, 128], [0, 128]]],
        shape_type='polygon',
        face_color=[1, 1, 1, 0], edge_color=[0, 0.6, 1, 1],
        edge_width=2, name='selection polygon')
    selection_layer.mode="SELECT"
    napari.run()
    viewer.show(block=True)  # This ensures it waits until Napari is closed

    labels_layer = viewer.layers[0]
    shapes_layer = viewer.layers[1]
    coords = np.round(shapes_layer.data[0]).astype(int)

    mask_2D = shapes_layer.to_masks().squeeze()
    mask_2D_img = np.copy(img)[:, :mask_2D.shape[0], :mask_2D.shape[1]]
    # broadcast the mask to the shape of the cropped image
    masking = np.broadcast_to(mask_2D, mask_2D_img.shape)
    mask_2D_img[~masking] = 0
    labels = np.unique(mask_2D_img)

    new_img = np.zeros_like(img)
    stats = regionprops(img)

    for i in range(len(stats)):
        if stats[i].label in labels:
            bbox = stats[i].bbox
            z_min, y_min, x_min, z_max, y_max, x_max = bbox
            roi = img[z_min:z_max, y_min:y_max, x_min:x_max]
            mask = (roi == stats[i].label)
            new_img[z_min:z_max,y_min:y_max, x_min:x_max][mask] = roi[mask]

    print("Select a region in Napari and close the viewer when done.")
    
    return new_img

#-----------------------------------------------------------------------------------------------------------------------

def tissue_mask(mask, radius=20):
    mgpu = cle.push_zyx(mask)
    ext_mgpu = cle.extend_labels_with_maximum_radius(mgpu, radius = radius)
    ext_img = closing(cle.pull(ext_mgpu), cube(100))
    ext_img = binary_fill_holes(ext_img).astype(int)

    components, num = label(ext_img, return_num=True, connectivity=3)

    propsa = regionprops(components)
    labels = [label.label for label in propsa]
    volumes = [label.area for label in propsa]

    max_id = np.argmax(volumes)
    max_mask = np.where(components == labels[max_id], 1, 0)
    max_vol = volumes[max_id]

    return max_mask, max_vol

#-----------------------------------------------------------------------------------------------------------------------

def eliminate_unwanted_objects(img):
    points_layer = define_points(img)
    points = points_layer.data.astype(int)
    unwanted_labels = img[points[:, 0], points[:, 1], points[:,2]]

    new_img = img.copy()
    stats = regionprops(new_img)

    for i in range(len(stats)):
        if stats[i].label in unwanted_labels:
            bbox = stats[i].bbox
            z_min, y_min, x_min, z_max, y_max, x_max = bbox
            roi = img[z_min:z_max, y_min:y_max, x_min:x_max]

            mask = (roi == stats[i].label)
            new_img[z_min:z_max,y_min:y_max, x_min:x_max][mask] = 0
    return new_img

#-----------------------------------------------------------------------------------------------------------------------

def define_points(img):

    # displai img
    viewer = napari.Viewer()
    viewer.add_image(img, name='image',  colormap='gist_earth')
    viewer.show(block=True)  # This ensures it waits until Napari is closed

    # choose points
    points_layer = viewer.add_points(np.empty((0, 3)), name='select_points', ndim=3)
    napari.run()
    
    return points_layer

#-----------------------------------------------------------------------------------------------------------------------


def define_volume(img, shape_size):
    # display rotated image and define coordinates for cropping in xy
    coords_xy = display_for_crop(img, "XY", shape_size=shape_size)
    [ymin, ymax, xmin, xmax] = extract_and_prep_roi(coords_xy, img.shape[1], img.shape[2])

    #choose z
    z_step = shape_size[0]//2
    points_layer = define_points(img)
    points = points_layer.data.astype(int)
    zmin, zmax = points[0][0]-z_step, points[0][0]+z_step


    print("rectangle coords", zmin, zmax, ymin, ymax, xmin, xmax)
    print(img.dtype)
    roi = img[zmin: zmax, ymin: ymax, xmin: xmax]

    return roi


#-----------------------------------------------------------------------------------------------------------------------


def eliminate_unwanted_objects(img):
    points_layer = define_points(img)
    points = points_layer.data.astype(int)
    unwanted_labels = img[points[:, 0], points[:, 1], points[:,2]]

    new_img = img.copy()
    stats = regionprops(new_img)

    for i in range(len(stats)):
        if stats[i].label in unwanted_labels:
            bbox = stats[i].bbox
            z_min, y_min, x_min, z_max, y_max, x_max = bbox
            roi = img[z_min:z_max, y_min:y_max, x_min:x_max]

            mask = (roi == stats[i].label)
            new_img[z_min:z_max,y_min:y_max, x_min:x_max][mask] = 0
    
    return new_img



#-----------------------------------------------------------------------------------------------------------------------

def save_img(img, filepath, resolution):
    img = np.array(img, dtype='uint16')
    imwrite(filepath, img, imagej=True,
            resolution=(1/resolution[2], 1/resolution[1]),
            metadata={'spacing': resolution[0],'axes': 'ZYX','unit': 'um'})
    return

#-----------------------------------------------------------------------------------------------------------------------
