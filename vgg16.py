from keras.engine.topology import Layer
from keras import backend as K
from keras import layers
from keras.models import Model
from keras.applications.imagenet_utils import preprocess_input
import tensorflow as tf
import numpy as np
import keras
import tensorflow as tf


class InputLayer(Layer):
    """
    Trainable input layer used for keras VGG16 model

    This layer will be used as trainable for input layer when creating VGG16 model in keras
    """
    def __init__(self, image_shape, **kwargs):
        """
        Initialize the layer

        Parameters:
        x_shape: shape of the picture to be generated, content image and style image
        """
        self.image_shape = image_shape
        super(InputLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Create a trainable weight variable for this layer.

        Parameters:
        input_shape: shape of the input tensor
        Weights:
        self.x_image: the image to be trained
        """
        self.image = self.add_weight(name='image', 
                                      shape=(self.image_shape),
                                      initializer='random_uniform',
                                      trainable=True)

        super(InputLayer, self).build(input_shape)

    def call(self, inputs):
        """
        Do nothing, just return the  weights

        This is a identity layer so we just return the weights unchanged
        Output: the tensor of shape (image_shape)
        """
        return preprocess_input(K.expand_dims(self.image, axis=0))
    
    def compute_output_shape(self, input_shape):
        return (1, self.image_shape[0], self.image_shape[1], self.image_shape[2])


def vgg16(image_shape):
    """
    Redefine the vgg16 from keras library

    Add a new layer before the original VGG16 from keras.
    This layer is used to pass the trainable image into our cnn network
    Parameters:
    image_shape: shape of the image
    Output:
    Modified keras model
    """
    inputs = layers.Input(shape=(1,))
    img_input = InputLayer(image_shape, name="block_input")(inputs)
    # Block 1
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(img_input)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    return Model(inputs=inputs, outputs = x)

def get_tf_vgg16_session(image_shape=(224,224,3), path_to_weights_file="./weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"):
    """
    Get the tensorflow graph of the keras redefined vgg16

    Parameters:
    image_shape: shape of the image to be trained
    Return:
    session that keras uses
    """
    model = vgg16(image_shape).load_weights(path_to_weights_file, by_name=True)
    session = K.get_session()
    return session
    
