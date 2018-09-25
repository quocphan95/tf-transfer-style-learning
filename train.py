import sys
import tensorflow as tf
import numpy as np
import vgg16
import preprocessor as pre
import cv2
import numpy as np

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
    m, n_H, n_W, n_C = tensor_content_layer_feature.get_shape().as_list()
    return 1 / (4 * n_H * n_W * n_C) * tf.reduce_sum(
        tf.square(tf.subtract(tensor_image_layer_feature, tensor_content_layer_feature))
        )
    
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

    def calculate_layer_error(style_layer_gram, image_layer_gram, layer_shape):
        m, n_H, n_W, n_C = layer_shape
        layer_error = (1 / (2 * n_H * n_W * n_C) ** 2 *
                       tf.reduce_sum(tf.square(tf.subtract(style_layer_gram, image_layer_gram))))
        return layer_error
        
    graph = session.graph
    if coeficients == None:
        coeficient = 1 / len(style_features)
        coeficients = {key: coeficient for key in style_features.keys()}
    jstyle = None
    for layer_name, style_layer_feature in style_features.items():
        style_layer_gram = get_gram_matrix(tf.constant(style_layer_feature, dtype=tf.float32))
        image_layer_gram = get_gram_matrix(graph.get_tensor_by_name(layer_name + "/Relu:0"))
        layer_error = coeficients[layer_name] * calculate_layer_error(style_layer_gram,
                                                                    image_layer_gram,
                                                                    style_layer_feature.shape)
        
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
    return (alpha * calculate_jcontent(session, content_features, content_layer_name) +
            beta * calculate_jstyle(session, style_features, coeficients))

if __name__ == "__main__":
    #assert len(sys.argv) > 2, "Not enough arguments"
    #content_path = sys.argv[1]
    #style_path = sys.argv[2]
    content_path = "./c.jpg"
    style_path = "./s.jpg"
    keras_session = vgg16.get_tf_vgg16_session()
    
    with keras_session as session:
        content_image = pre.read_image(content_path)
        content_features = get_all_features(session, content_image)
        style_image = pre.read_image(style_path)
        style_features = get_all_features(session, style_image)
        image_tensor = [var for var in tf.global_variables() if var.name=="block_input/image:0"][0]
        
        init_image = np.zeros((224,224,3)).astype(np.uint8) #(style_image * 0.4 + content_image * 0.6).astype(np.uint8)
        cv2.imshow("image", init_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        session.run(tf.assign(image_tensor, init_image))
        
        #j = calculate_j(session, content_features, style_features, "block5_conv1", alpha=10, beta=40)
        j = calculate_jcontent(session, content_features, "block1_conv1")
        #j = calculate_jstyle(session, style_features)
        
        optimizer = tf.train.AdamOptimizer(10)
        minimize_op = optimizer.minimize(j, var_list=[image_tensor])
        session.run(tf.variables_initializer(optimizer.variables()))
        stop_training = False
        #while not stop_training:
        for iteration in range(140):
            try:
                (cost, _) = session.run((j, minimize_op))
                print("{0:>6}:{1:>15}".format(iteration, cost))
            except KeyboardInterrupt:
                stop_training = True
                break
        generated_image = session.run(image_tensor).astype(np.uint8)
        cv2.imshow("image", generated_image)
        cv2.imwrite("out.jpg", generated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
            
        
    
    

    
    
    
    
