import sys
import tensorflow as tf
import numpy as np
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
    tensor_image_layer_feature = graph.get_tensor_by_name(layer_name + "/Relu:0")
    tensor_content_layer_feature = tf.constant(content_features[layer_name], dtype=tf.float32)
    return tf.losses.mean_squared_error(tensor_content_layer_feature, tensor_image_layer_feature, 0.5)
    
def calculate_jstyle(session, style_features, coeficients=None):
    """
    Calculate the jstyle of the image base on features of style image

    Parameters:
    session: the session contains tf graph of the convnet
    style_features: a dictionary contains all features of style image
    coeficients: a dictionary contains coeficients associate with the layers
    """
    def get_gram_matrix(feature):
        flat_feature = tf.reshape(feature, (-1, feature.shape[-1]))
        return tf.matmul(tf.transpose(flat_feature), flat_feature)
        
    graph = session.graph
    if coeficients == None:
        coeficient = 1 / len(style_features)
        coeficients = {key: coeficient for key in style_features.keys()}
    jstyle = None
    for layer_name, style_layer_feature in style_features.items():
        tensor_style_layer_feature = tf.constant(style_layer_feature, dtype=tf.float32)
        tensor_image_layer_feature = graph.get_tensor_by_name(layer_name + "/Relu:0")
        style_layer_gram = get_gram_matrix(tensor_style_layer_feature)
        image_layer_gram = get_gram_matrix(tensor_image_layer_feature)
        feature_width = style_layer_feature.shape[-2]
        feature_height = style_layer_feature.shape[-2]
        layer_error = tf.losses.mean_squared_error(style_layer_gram,
                                                   image_layer_gram,
                                                   coeficients[layer_name]/ (4 * feature_width**2 * feature_height**2))
        jstyle = layer_error if jstyle == None else tf.add(jstyle, layer_error)
    return jstyle
            
    
def calculate_j(session,
                content_features, style_features,
                content_layer_name, coeficients=None,
                alpha = 0.5, beta = 0.5):
    """
    Calculate the loss function
    
    Parameters:
    session: the session contains tf graph of the convnet
    content_features: a dictionary contains all features of content image
    layer_name: name of the layer used to calculate jcontent
    style_features: a dictionary contains all features of style image
    coeficients: a dictionary contains coeficients associate with the layers
    alpha: alpha coeficient
    beta: beta coeficient
    Return:
    A tensor represent the loss
    """
    return alpha * calculate_jcontent(session, content_features) + beta * calculate_jstyle(session, style_feature, coeficients)

if __name__ == "__main__":
    #content_path = sys.argv[1]
    #style_path = sys.argv[2]
    session = vgg16.get_tf_vgg16_session()
    content_features = get_all_features(session, np.zeros((224,224,3)))
    style_features = get_all_features(session, np.zeros((224,224,3)))
    print(calculate_jstyle(session, content_features))
    
    
    
    
    
