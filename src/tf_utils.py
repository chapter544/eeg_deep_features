import tensorflow as tf
import numpy as np

def PReLU(_x, name="prelu"):
    _alpha = tf.get_variable(name, shape=_x.get_shape()[-1],
            dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
    return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)


def build_fc_layer(X, in_dim, out_dim, layer_name, is_training,
        use_dropout=False, keep_prob=0.5,
        use_BN=False, use_BN_Contrib=False, 
        use_BN_Front=False,
        activation='relu'):

    stddev = np.sqrt(2.0/in_dim)
    with tf.variable_scope(layer_name):
        W = tf.get_variable(
                initializer=tf.truncated_normal(
                    shape=[in_dim, out_dim], stddev=stddev), name='W')
        b = tf.get_variable(
                initializer=tf.truncated_normal(
                    shape=[out_dim], stddev=stddev), name='b')
        h = tf.nn.bias_add(tf.matmul(X, W), b)

        if use_BN is True: # use BN layer
            if use_BN_Front is True: # use BN in front of activation
                if use_BN_Contrib is True:
                    h = batch_norm_contrib(h, is_training)
                else:
                    h = tf.contrib.layers.batch_norm(
                                h, 
                                is_training=is_training, 
                                decay=0.99,
                                center=True, 
                                scale=True, 
                                activation_fn=tf.nn.relu, 
                                updates_collections=None) 

                if activation == 'elu':
                    h = tf.nn.elu(h, name=layer_name)
                elif activation == 'prelu':
                    h = PReLut(h, name=layer_name)
                else:
                    h = tf.nn.relu(h, name=layer_name)
            else: # use BN after activation
                if activation == 'elu':
                    h = tf.nn.elu(h, name=layer_name)
                elif activation == 'prelu':
                    h = PReLut(h, name=layer_name)
                else:
                    h = tf.nn.relu(h, name=layer_name)

                if use_BN_Contrib is True:
                    h = batch_norm_contrib(h, is_training)
                else:
                    h = tf.contrib.layers.batch_norm(
                                h, 
                                is_training=is_training, 
                                decay=0.99,
                                center=True, 
                                scale=True, 
                                activation_fn=tf.nn.relu, 
                                updates_collections=None) 

        else: # do not use BN
            if activation == 'elu':
                h = tf.nn.elu(h, name=layer_name)
            elif activation == 'prelu':
                h = PReLut(h, name=layer_name)
            elif activation == 'relu':
                h = tf.nn.relu(h, name=layer_name)

        if use_dropout is True:
            h = tf.nn.dropout(h, keep_prob)

        return h






def weight_variable(shape):
    #initial = tf.truncated_normal(shape, stddev=0.01)
    initial = tf.truncated_normal(shape, stddev=0.03)
    return tf.Variable(initial)

def bias_variable(shape): 
    initial = tf.constant(0.00, shape=shape)
    return tf.Variable(initial)

def batch_norm_contrib(x, is_training):
    h = tf.contrib.layers.batch_norm(x, 
                    center=True, scale=True, 
                    is_training=is_training)
    return h



def fc_layer(layer_name, input_data, in_dim, out_dim, activation='elu'):
    with tf.variable_scope(layer_name):
        W = tf.get_variable('W', initializer=tf.truncated_normal([in_dim,
            out_dim], stddev=0.01))
        b = tf.get_variable('b', initializer=tf.zeros([out_dim]))
        pre_activation = tf.matmul(input_data, W) + b

        if activation == 'relu':
            layer = tf.nn.relu(pre_activate)
        elif activation == 'sigmoid':
            layer = tf.nn.sigmoid(pre_activate)
        else:
            layer = tf.nn.elu(pre_activate)
    return layer

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
                #h = batch_norm_contrib(h, is_training)
                h = batch_norm_wrapper(h, is_training)

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
                out = batch_norm_wrapper(h_act, is_training)
    return out



def batch_norm(x, scope, is_training_phase, epsilon=0.001, decay=0.99):
    """
    Returns a batch normalization layer that automatically switch between train and test phases based on the 
    tensor is_training

    Args:
        x: input tensor
        scope: scope name
        is_training: boolean tensor or variable
        epsilon: epsilon parameter - see batch_norm_layer
        decay: epsilon parameter - see batch_norm_layer

    Returns:
        The correct batch normalization layer based on the value of is_training
    """
    outputs = tf.contrib.layers.batch_norm(
			x, 
			is_training=is_training_phase, 
			decay=0.9,
			center=True, 
			scale=True, 
			activation_fn=tf.nn.relu, 
			updates_collections=None, 
			scope=scope),
    return outputs

def batch_norm_wrapper(inputs, is_training, epsilon=1e-6, decay=0.9):
    scale = tf.get_variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.get_variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.get_variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.get_variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

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
