"""Implements feature extraction and other data processing helpers.
(This file will not be graded).
"""

import numpy as np
import skimage
from skimage import color


def preprocess_data(data, process_method='default'):
    """Preprocesses dataset.

    Args:
        data(dict): Python dict loaded using io_tools.
        process_method(str): processing methods needs to support
          ['raw', 'default'].
        if process_method is 'raw'
          1. Convert the images to range of [0, 1]
          2. Remove dataset mean. Average the images across the batch dimension.
             This will result in a mean image of dimension (8,8,3).
          3. Flatten images, data['image'] is converted to dimension (N, 8*8*3)
        if process_method is 'default':
          1. Convert images to range [0, 1]
          2. Convert from rgb to gray then back to rgb. Use skimage
          3. Take the absolute value of the difference with the original image.
          4. Remove dataset mean. Average the absolute value differences across
             the batch dimension. This will result in a mean of dimension (8,8,3).
          5. Flatten images, data['image'] is converted to dimension (N, 8*8*3)

    Returns:
        data(dict): Apply the described processing based on the process_method
        str to data['image'], then return data.
    """
    if process_method == 'raw':
        convert_image_to_float(data)
        remove_data_mean(data)
        flatten_image(data)

    elif process_method == 'default':
        convert_image_to_float(data)
        original_images = data['image']
        convert_image_to_grayscale(data)
        convert_image_to_rgb(data)
        convert_image_abs_difference(original_images, data)
        remove_data_mean(data)
        flatten_image(data)

    elif process_method == 'custom':
        # Design your own feature!
        pass

    return data

def convert_image_abs_difference(original_images, data):
    images = data['image']
    abs_diff_images = np.absolute(original_images - images)
    data['image'] = abs_diff_images
    return data

def convert_image_to_grayscale(data):
    images = data['image']
    grayscale_images = []
    for image in images:
        grayscale_images.append(skimage.color.rgb2gray(image))
    data['image'] = np.array(grayscale_images)
    return data

def convert_image_to_rgb(data):
    images = data['image']
    rgb_images = []
    for image in images:
        rgb_images.append(skimage.color.gray2rgb(image))
    data['image'] = np.array(rgb_images)
    return data

def flatten_image(data):
    images = data['image']
    N, W, H, C = images.shape
    flattened_images = np.reshape(images, (N, W*H*C))
    data['image'] = flattened_images
    return data

def convert_image_to_float(data):
    images = data['image']
    float_images = images / 255
    data['image'] = float_images
    return data

def compute_image_mean(data):
    """ Computes mean image.

    Args:
        data(dict): Python dict loaded using io_tools.

    Returns:
        image_mean(numpy.ndarray): Average across the example dimension.
    """
    images = data['image']
    return np.mean(images, axis=0)

def remove_data_mean(data):
    """Removes data mean.

    Args:
        data(dict): Python dict loaded using io_tools.

    Returns:
        data(dict): Remove mean from data['image'] and return data.
    """
    images = data['image']
    centered_images = images - compute_image_mean(data)
    data['image'] = centered_images
    return data
