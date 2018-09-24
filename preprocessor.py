import cv2

def read_image(image_path, output_size = (224, 224)):
    """
    Read and preprocess an image

    Parameters:
    image_path: path to the image to be read
    output_size: the size of the image to be resized to
    """
    image = cv2.imread(image_path)
    image = cv2.resize(image, dsize=output_size, interpolation=cv2.INTER_CUBIC)
    return image
