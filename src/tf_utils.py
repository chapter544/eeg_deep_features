import tensorflow as tf
import numpy as np

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    #initial = tf.truncated_normal(shape, stddev=0.05)
    return tf.Variable(initial)

def bias_variable(shape): 
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


#def conv2d(x, W, strides=strides, padding=padding):
#    return tf.nn.conv2d(x, W, strides=strides, padding=padding)

#def max_pool(x, ksize=ksize, strides=strides, padding=padding):
#    return tf.nn.max_pool(x, ksize, strides, padding=padding)


def dense_layer(X, 
        dims, 
        scope, 
        activation='elu', 
        use_bn=False, bn_first=True,
        is_training=False):
    with tf.name_scope(scope):
        in_dim, out_dim = dims
        W = weight_variable([in_dim, out_dim])
        b = weight_variable([out_dim])
        h = tf.matmul(X, W) + b

        if bn_first == True:
            if use_bn == True:
                h = batch_norm_contrib(h, is_training)

            if activation == 'elu':
                out = tf.nn.elu(h)
            else:
                out = tf.nn.relu(h)
        else:
            if activation == 'elu':
                h_act = tf.nn.elu(h)
            else:
                h_act = tf.nn.elu(h)

            if use_bn == True:
                out = batch_norm_contrib(h_act, is_training)
    return out






def batch_norm_wrapper(inputs, is_training, decay=0.999):
    epsilon = 1e-6
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training is True:
        batch_mean, batch_var = tf.nn.moments(inputs, [0])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                               pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs, batch_mean, batch_var,
                                             beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs, pop_mean, pop_var,
                                         beta, scale, epsilon)
