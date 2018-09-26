import sys
import tensorflow as tf
import numpy as np
import vgg16
import preprocessor as pre
import cv2
import numpy as np

class CONFIG:
    image_shape = (224, 224, 3)
    style_coefs = (
            ("block1_conv1", 0.2),
            ("block2_conv1", 0.2),
            ("block3_conv1", 0.2),
            ("block4_conv1", 0.2),
            ("block5_conv1", 0.9))
    content_layer = "block1_conv1"
    conv_layer_names = (
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
        "block5_conv3")

def get_all_features(session, image, conv_layer_names):
    """
    Compute the outputs of all conv layers

    Parameters:
    session: the session contains tf graph of the convnet
    inmage: numpy array contain the content of the image to be computed
    conv_layer_names: names of all conv layers
    Return:
    A dictionary contains outputs of all conv layers
    """
    
    features = {}
    graph = session.graph
    for conv in conv_layer_names:
        tensor_name = conv + "/Relu:0"
        tensor = graph.get_tensor_by_name(tensor_name)
        features[conv] = tensor
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
    tensor_content_layer_feature = tf.constant(content_features[layer_name])
    m, n_H, n_W, n_C = content_features[layer_name].shape
    return 1 / (4 * n_H * n_W * n_C) * tf.reduce_sum(
        tf.square(tf.subtract(tensor_image_layer_feature, tensor_content_layer_feature)))
    
def calculate_jstyle(session, style_features, coeficients):
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

    def calculate_layer_error(style_layer_gram, image_layer_gram, layer_shape):
        m, n_H, n_W, n_C = layer_shape
        layer_error = (1 / (2 * n_H * n_W * n_C) ** 2 *
                       tf.reduce_sum(tf.square(tf.subtract(style_layer_gram, image_layer_gram))))
        return layer_error
        
    graph = session.graph
    jstyle = 0
    for layer_name, coeficient in coeficients:
        style_layer_gram = get_gram_matrix(tf.constant(style_features[layer_name], dtype=tf.float32))
        image_layer_gram = get_gram_matrix(graph.get_tensor_by_name(layer_name + "/Relu:0"))
        layer_error = coeficient * calculate_layer_error(style_layer_gram, image_layer_gram, style_features[layer_name].shape)
        jstyle += layer_error
    return jstyle
            
    
def calculate_j(jcontent, jstyle, alpha = 10, beta = 40):
    """
    Calculate the loss function
    
    Parameters:
    jcontent: content error
    jstyle: style error
    alpha: alpha coeficient
    beta: beta coeficient
    Return:
    A tensor represent the loss
    """
    return alpha * jcontent + beta * jstyle

if __name__ == "__main__":
    #assert len(sys.argv) > 2, "Not enough arguments"
    #content_path = sys.argv[1]
    #style_path = sys.argv[2]
    content_path = "./c.jpg"
    style_path = "./s.jpg"
    keras_session = vgg16.get_tf_vgg16_session()
    
    with keras_session as session:
        content_image = pre.read_image(content_path)
        content_features = get_all_features(session, content_image, CONFIG.conv_layer_names)
        style_image = pre.read_image(style_path)
        style_features = get_all_features(session, style_image, CONFIG.conv_layer_names)
        image_tensor = [var for var in tf.global_variables() if var.name=="block_input/image:0"][0]
        
        init_image = np.zeros((224,224,3)).astype(np.uint8)
        
        jcontent = calculate_jcontent(session, content_features, CONFIG.content_layer)
        jstyle = calculate_jstyle(session, style_features, CONFIG.style_coefs)
        j = calculate_j(jcontent, jstyle)

        j = jcontent
        
        optimizer = tf.train.AdamOptimizer(10)
        minimize_op = optimizer.minimize(j, var_list=[image_tensor])
        session.run(tf.variables_initializer(optimizer.variables()))
        stop_training = False
        iteration = 1
        while not stop_training:
            try:
                (cost, _) = session.run((j, minimize_op))
                print("{0:>6}:{1:>15}".format(iteration, cost))
                iteration += 1
            except KeyboardInterrupt:
                stop_training = True
                break
        generated_image = session.run(image_tensor).astype(np.uint8)
        cv2.imshow("image", generated_image)
        cv2.imwrite("out.jpg", generated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
            
        
    
    

    
    
    
    
