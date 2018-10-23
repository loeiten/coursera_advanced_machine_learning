# Original Version: Taehoon Kim (http://carpedm20.github.io)
#   + Source: https://github.com/carpedm20/DCGAN-tensorflow/blob/e30539fb5e20d5a0fed40935853da97e9e55eee8/utils.py
#   + License: MIT

import numpy as np
import cv2

def get_image(image_path, image_size, is_crop=True):
    """
    Returns a cropped image
    
    Parameters
    ----------
    image_path : Path or str
        The image to read
    image_size : int
        The size of the image
    is_crop : bool
        Whether or not the image should be cropped
        
    Returns
    -------
    np.array, shape (image_size, image_size, channels)
        The transformed image (BGR)
    """
    return transform(imread(image_path), image_size, is_crop)


def save_images(images, size, image_path):
    """
    Wrapper to imsave function
    
    Parameters
    ----------
    images : np.array (n_img, width, heigth, channels)
        The images to be saved
    size : array-like, shape (2,)
        Rows and cols of the resulting image
    path : Path or str
        Save path om the image
    """
    imsave(images, size, image_path)


def imread(path):
    """
    Read an image
    
    Parameters
    ----------
    path : Path or str
        Path to the image
        
    Returns
    -------
    np.array, shape (width, height, channels)
        The image as an np.array (BGR)
    """
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def merge_images(images, size):
    """
    Transforms and merges images
    
    Parameters
    ----------
    images : np.array (n_img, width, heigth, 3)
        The images to be merged
    size : array-like, shape (2,)
        Rows and cols of the resulting image
    
    Returns
    --------
    img : np.array (size[0], szie[1], 3)
        The merged images
    """
    return inverse_transform(merge(images, size))


def merge(images, size):
    """
    Merges several images to one
    
    Parameters
    ----------
    images : np.array (n_img, width, heigth, 3)
        The images to be merged
    size : array-like, shape (2,)
        Rows and cols of the resulting image
        
    Returns
    -------
    img : np.array (size[0], szie[1], 3)
        The merged images
    """
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((int(h * size[0]), int(w * size[1]), 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return img


def imsave(images, size, path):
    """
    Merges and saves the images
    
    Parameters
    ----------
    images : np.array (n_img, width, heigth, channels)
        The images to be saved
    size : array-like, shape (2,)
        Rows and cols of the resulting image
    path : Path or str
        Save path om the image
    """
    img = merge(images, size)
    cv2.imwrite(str(path), (255*inverse_transform(img)).astype(np.uint8))


def transform(image, npx=64, is_crop=True):
    """
    Transforms the image to be used in a GAN
    
    Specifically it will normalize the image to be within the range [-1, 1]
    
    Parameters
    ----------
    image : np.array (width, height, channels)
        The image to transform
    npx : int
        The final size of the image
    is_crop: bool
        Whether or not to crop
    
    Returns
    -------
    np.array, shape (npx, npx, channels)
        The transformed image
    """
    if is_crop:
        cropped_image = center_crop(image, npx)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.


def center_crop(img, crop_h, crop_w=None):
    """
    Crop the center of an image
    
    Note
    ----
    In order to capture faces, the crop values are first doubled, then the resulting image in resized
    
    Parameters
    ----------
    img : np.array, shape (width, height, channels)
        The image to be cropped
    crop_h : int
        The heigth crop
        If crop_w is None, then crop_w = crop_h
    crop_w : None or int
        The weigth crop
        If crop_w is None, then crop_w = crop_h 
       
    Returns
    -------
    cropped : np.array, shape (crop_h, crop_w, channels)
        The cropped image
    
    References
    ----------
    https://www.coursera.org/learn/deep-learning-in-computer-vision/discussions/weeks/5/threads/oCpC6JmzEeiylwrzdYR6Ug
    https://github.com/carpedm20/DCGAN-tensorflow/blob/e30539fb5e20d5a0fed40935853da97e9e55eee8/utils.py
    """
    
    if crop_w is None:
        crop_w = crop_h
    h, w = img.shape[:2]

    j = h//2 - crop_h
    i = w//2 - crop_w
    cropped = cv2.resize(img[j:j+crop_h*2, i:i+crop_w*2], (crop_h, crop_w))
    
    return cropped


def inverse_transform(images):
    """
    Rescales the image to have values between [0, 1]
    
    Parameters
    ----------
    images : np.array (n_img, width, heigth, channels)
        The images to be rescaled
        
    Returns
    -------
    np.array, shape (n_img, width, heigth, channels)
        The rescaled images
    """
    return (images+1.)/2.