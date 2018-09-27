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
    decay_rate = 0.75 # decrease lr by 25% when optimizer overshoot
    freeze_learing_rate = 40
    standard_precision = 6
    init_learning_rate = 10.0

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
    def get_gram_matrix_tensor(feature):
        assert isinstance(feature, tf.Tensor), "input must be a tensor"
        flat_feature = tf.reshape(feature, (-1, feature.shape[-1]))
        return tf.matmul(tf.transpose(flat_feature), flat_feature)

    def get_gram_matrix(feature):
        assert isinstance(feature, np.ndarray), "input must be numpy array"
        flat_feature = feature.reshape((-1, feature.shape[-1]))
        return np.dot(flat_feature.T, flat_feature)

    def calculate_layer_error(style_layer_gram, image_layer_gram, layer_shape):
        m, n_H, n_W, n_C = layer_shape
        layer_error = (1 / (2 * n_H * n_W * n_C) ** 2 *
                       tf.reduce_sum(tf.square(tf.subtract(style_layer_gram, image_layer_gram))))
        return layer_error
        
    graph = session.graph
    jstyle = 0
    for layer_name, coeficient in coeficients:
        style_layer_gram = get_gram_matrix(style_features[layer_name])
        image_layer_gram = get_gram_matrix_tensor(graph.get_tensor_by_name(layer_name + "/Relu:0"))
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

def show_image(title, image):
    """
    Show an image

    Parameters:
    title: title of the window that show the image
    image: image to be shown
    """
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def generate_init_image(content, style, noise_rate=0.2):
    """
    Generate an intit image for training based on content and style images

    Parameters:
    content: the content image that was not preprocessed
    style: the style image that was not preprocessed
    noise_rate: the noise rate
    Return:
    the init image ready for training
    """
    shape = content.shape
    assert shape == style.shape, "content and style shape must be the same"
    noise_image = np.random.uniform(0, 255, shape)
    image = (1 - noise_rate) * (content) + noise_rate * noise_image
    return image.astype(np.uint8)


if __name__ == "__main__":
    #assert len(sys.argv) > 2, "Not enough arguments"
    #content_path = sys.argv[1]
    #style_path = sys.argv[2]
    content_path = "./c.jpg"
    style_path = "./s.jpg"
    keras_session = vgg16.get_tf_vgg16_session()
    
    with keras_session as session:
        content_image = pre.read_image(content_path, CONFIG.image_shape[0:2])
        style_image = pre.read_image(style_path, CONFIG.image_shape[0:2])
        init_image = generate_init_image(content_image, style_image)
        show_image("", init_image)
        content_image = pre.preprocess_image(content_image)
        style_image = pre.preprocess_image(style_image)
        content_features = get_all_features(session, content_image, CONFIG.conv_layer_names)
        style_features = get_all_features(session, style_image, CONFIG.conv_layer_names)
        image_tensor = [var for var in tf.global_variables() if var.name=="block_input/image:0"][0]
        init_image = pre.preprocess_image(init_image)
        session.run(tf.assign(image_tensor, init_image))
        jcontent = calculate_jcontent(session, content_features, CONFIG.content_layer)
        jstyle = calculate_jstyle(session, style_features, CONFIG.style_coefs)
        j = calculate_j(jcontent, jstyle)
        learning_rate = CONFIG.init_learning_rate
        learning_rate_ph = tf.placeholder(dtype=tf.float32)
        precision = CONFIG.standard_precision # number of decimal places
        optimizer = tf.train.AdamOptimizer(learning_rate_ph)
        minimize_op = optimizer.minimize(j, var_list=[image_tensor])
        session.run(tf.variables_initializer(optimizer.variables()))
        stop_training = False
        iteration = 1
        cost = np.inf
        last_cost = np.inf
        freeze_iteration_remain = CONFIG.freeze_learing_rate # Wait at least some iters before changing learning rate again
        while not stop_training:
            try:
                last_cost = cost
                (cost, _) = session.run((j, minimize_op), feed_dict={learning_rate_ph: learning_rate})
                print("{0:>6}:{1:>40.{2}f}".format(iteration, cost, precision))
                iteration += 1
                freeze_iteration_remain -= freeze_iteration_remain and 1
                # learinig rate decay
                if not (cost < last_cost) and not freeze_iteration_remain: # overshooting happen
                    learning_rate = learning_rate * CONFIG.decay_rate
                    freeze_iteration_remain = CONFIG.freeze_learing_rate
                    print("Overshooting happen!, learing rate now is {0:.{1}f}".format(learning_rate, CONFIG.standard_precision))
            except KeyboardInterrupt:
                command = input("\ncommand ({0:.{1}f}):".format(learning_rate, CONFIG.standard_precision))
                if command == "stop":
                    stop_training = True
                elif command == "show":
                    generated_image = pre.depreprocess_image(session.run(image_tensor)).astype(np.uint8)
                    show_image("generated image", generated_image)
                elif command == "save":
                    generated_image = pre.depreprocess_image(session.run(image_tensor)).astype(np.uint8)
                    cv2.imwrite("out.jpg", generated_image)
                elif command.startswith("precision "):
                    try:
                        precision = int(command.split()[1])
                    except (ValueError, IndexError):
                        pass
        generated_image = pre.depreprocess_image(session.run(image_tensor)).astype(np.uint8)
        show_image("generated image", generated_image)
        cv2.imwrite("out.jpg", generated_image)
            
        
    
    

    
    
    
    
