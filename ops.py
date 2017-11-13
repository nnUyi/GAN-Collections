import tensorflow as tf
import numpy as np
import scipy.misc
import math

# convolution
def conv2d(input_x, kernel_size, stride=[1,2,2,1], scope_name='conv2d', conv_type='SAME'):
    output_len = kernel_size[3]
    with tf.variable_scope(scope_name):
        weights = tf.get_variable('weights', kernel_size, tf.float32, initializer=tf.random_normal_initializer(stddev=0.02))
        bias = tf.get_variable('bias', output_len, tf.float32, initializer=tf.constant_initializer(0))
        conv = tf.nn.bias_add(tf.nn.conv2d(input_x, weights, strides=stride, padding=conv_type), bias)
        return(conv)

# deconvolution
def deconv2d(input_x, kernel_size, output_shape, stride=[1,2,2,1], scope_name='deconv2d', deconv_type='SAME'):
    output_len = kernel_size[2]
    with tf.variable_scope(scope_name):
        weights = tf.get_variable('weights', kernel_size, tf.float32, initializer=tf.random_normal_initializer(stddev=0.02))
        bias = tf.get_variable('bias', output_len, tf.float32, initializer=tf.constant_initializer(0))
        try:
            deconv = tf.nn.bias_add(tf.nn.conv2d_transpose(input_x, weights, output_shape, strides=stride, padding=deconv_type), bias)
        except:
            deconv = tf.nn.bias_add(tf.nn.deconv2d(input_x, weights, output_shape, strides=stride, padding=deconv_type), bias)
        return deconv

# batch normalization
def batch_norm(input_x, epsilon=1e-5, momentum=0.9, is_training = True, name='batch_name'):
    with tf.variable_scope(name) as scope:
        batch_normalization = tf.contrib.layers.batch_norm(input_x,
                                              decay=momentum,
                                              updates_collections=None,
                                              epsilon=epsilon,
                                              scale=True,
                                              is_training=is_training,
                                              scope=name)
        return batch_normalization
        
# fully connected
def linear(input_x, output_size, scope_name='linear'):
    shape = input_x.get_shape()
    input_size = shape[1]
    with tf.variable_scope(scope_name):
        weights = tf.get_variable('weights', [input_size, output_size], tf.float32, initializer=tf.random_normal_initializer(stddev=0.02))
        bias = tf.get_variable('bias', output_size, tf.float32, initializer=tf.constant_initializer(0))        
        output = tf.matmul(input_x, weights) + bias
        return output

# leaky_relu
def leaky_relu(input_x, leaky=0.2):
    return tf.maximum(leaky*input_x, input_x)

# pooling
def max_pool(input_data_x, filter_shape=[1,2,2,1], pooling_type='SAME'):
    if pooling_type == 'SAME':
        return tf.nn.max_pool(input_data_x, ksize=filter_shape, strides=[1,2,2,1], padding=pooling_type)
    else:
        return tf.nn.max_pool(input_data_x, ksize=filter_shape, strides=[1,2,2,1], padding=pooling_type)

# conv out size
def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))

# load data from datasets
def get_image(batch_file, is_grayscale=False):
    if is_grayscale:
        return scipy.misc.imread(batch_file, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(batch_file).astype(np.float)
        
def save_images(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return img
