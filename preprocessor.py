import cv2
import numpy as np

__all__ = ["read_image", "preprocess_image", "depreprocess_image"]

class CONFIG:
    mean = np.asarray([[123.68, 103.939, 116.779]])
    std_dev = np.asarray([[58.395, 57.12, 57.375]])

def read_image(image_path, output_size):
    """
    Read and preprocess an image

    Parameters:
    image_path: path to the image to be read
    output_size: the size of the image to be resized to
    """
    image = cv2.imread(image_path)
    if image.shape != output_size:
        image = cv2.resize(image, dsize=output_size, interpolation=cv2.INTER_CUBIC).astype(np.float32)
    return image

def preprocess_image(image):
    """
    Preprocess the image

    Subtract the image by VGG16 mean from imagenet
    Prarameters:
    image: numpy matrix represent
    Return:
    preprocessed image
    """
    return image - CONFIG.mean

def depreprocess_image(preprocessed_image):
    """
    Covert the preprocessed image to the original format

    Add the image with VGG16 mean from imagenet
    Parameters:
    preprocessed_image: image that was preprocessed
    Return:
    image in original format
    """
    return preprocessed_image + CONFIG.mean
