import tensorflow as tf
import numpy as np

def weight_variable(shape):
<<<<<<< HEAD
    initial = tf.truncated_normal(shape, stddev=0.5)
=======
    #initial = tf.truncated_normal(shape, stddev=0.01)
    initial = tf.truncated_normal(shape, stddev=0.03)
>>>>>>> a3ed0a30cbe6b19704b4556dc6cf862fcf9e321f
    return tf.Variable(initial)

def bias_variable(shape): 
    initial = tf.constant(0.5, shape=shape)
    return tf.Variable(initial)


#def conv2d(x, W, strides=strides, padding=padding):
#    return tf.nn.conv2d(x, W, strides=strides, padding=padding)

#def max_pool(x, ksize=ksize, strides=strides, padding=padding):
#    return tf.nn.max_pool(x, ksize, strides, padding=padding)


