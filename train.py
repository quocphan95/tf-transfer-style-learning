import sys
import tensorflow as tf
import numpy as np
import cv2
import vgg16

def get_all_features(session, image):
    """
    Compute the outputs of all conv layers

    Parameters:
    session: the session contains tf graph of the convnet
    inmage: numpy array contain the content of the image to be computed
    Return:
    A dictionary contains outputs of all conv layers
    """
    conv_layer_names = [
        "block1_conv1",
        "block1_conv2",
        "block2_conv1",
        "block2_conv2",
        "block3_conv1",
        "block3_conv2",
        "block3_conv3",
        "block4_conv1",
        "block4_conv2",
        "block4_conv3",
        "block5_conv1",
        "block5_conv2",
        "block5_conv3"]
    features = {}
    graph = session.graph
    for conv in conv_layer_names:
        tensor_name = conv + "/Relu:0"
        tensor = graph.get_tensor_by_name(tensor_name)
        features[conv] = tensor
    input_tensor = graph.get_tensor_by_name("input_1:0")
    image_tensor = [var for var in tf.global_variables() if var.name=="block_input/image:0"][0]
    session.run(tf.assign(image_tensor, image))
    return session.run(features)

def calculate_jcontent(session, content_features, layer_name):
    """
    Calculate the jcontent of the image base on a layer

    Parameters:
    session: the session contains tf graph of the convnet
    content_features: a dictionary contains all features of content image
    layer_name: name of the layer used to calculate jcontent
    Return:
    A tensor represents the jcontent loss
    """
    graph = session.graph
    F_tensor = graph.get_tensor_by_name(layer_name + "/Relu:0")
    P_tensor = tf.constant(content_features[layer_name], dtype=tf.float32)
    return tf.losses.mean_squared_error(P_tensor, F_tensor, 0.5)
    
def calculate_jstyle():
    """
    """
    raise NotImplementedError
def calculate_j():
    """
    """
    raise NotImplementedError

if __name__ == "__main__":
    content_path = sys.argv[1]
    style_path = sys.argv[2]
    
    session = vgg16.get_tf_vgg16_session()
    features = get_all_features(session, np.zeros((224,224,3)))
    
    
    
    
