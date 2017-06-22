# 07/11/2017
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import h5py
import glob
import os
import argparse
import sys
from datetime import datetime

FLAGS = None

def build_fc_big_freqFlatten(x, x_dim, keep_prob):
    # FC1
    with tf.name_scope("FC1"):
        fc1_dim = 300
        W_fc1 = weight_variable([x_dim, fc1_dim])
        b_fc1 = weight_variable([fc1_dim])
        h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

    # FC2
    with tf.name_scope("FC2"):
        fc2_dim = 200
        W_fc2 = weight_variable([fc1_dim, fc2_dim])
        b_fc2 = weight_variable([fc2_dim])
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    # FC3
    with tf.name_scope("FC3"):
        fc3_dim = 100 
        W_fc3 = weight_variable([fc2_dim, fc3_dim])
        b_fc3 = weight_variable([fc3_dim])
        h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)

    # dropout
    with tf.name_scope("Dropout"):
        h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)

    # FC4
    with tf.name_scope("FC4"):
        fc4_dim = 200
        W_fc4 = weight_variable([fc3_dim, fc4_dim])
        b_fc4 = weight_variable([fc4_dim])
        h_fc4 = tf.nn.relu(tf.matmul(h_fc3_drop, W_fc4) + b_fc4)

    # FC5
    with tf.name_scope("FC5"):
        fc5_dim = 300
        W_fc5 = weight_variable([fc4_dim, fc5_dim])
        b_fc5 = weight_variable([fc5_dim])
        h_fc5 = tf.nn.relu(tf.matmul(h_fc4, W_fc5) + b_fc5)

    # FC6
    with tf.name_scope("FC6"):
        fc6_dim = x_dim
        W_fc6 = weight_variable([fc5_dim, fc6_dim])
        b_fc6 = weight_variable([fc6_dim])
        y = tf.matmul(h_fc5, W_fc6) + b_fc6

    # LOSS 
    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.squared_difference(y, x))
        #l1_loss = tf.reduce_sum(tf.abs(W_fc1)) + \
        #          tf.reduce_sum(tf.abs(W_fc2)) + \
        #          tf.reduce_sum(tf.abs(W_fc3)) + \
        #          tf.reduce_sum(tf.abs(W_fc4)) +  \
        #          tf.reduce_sum(tf.abs(W_fc5)) +  \
        #          tf.reduce_sum(tf.abs(W_fc6))
        #loss += gamma * l1_loss
        tf.scalar_summary("loss", loss)

        # summary
        summary_op = tf.merge_all_summaries()
    return loss, y




def build_fc_freqFlatten_L1(x, x_dim, keep_prob, gamma=0.00001):
    # FC1
    with tf.name_scope("FC1"):
        fc1_dim = 256 
        W_fc1 = weight_variable([x_dim, fc1_dim])
        b_fc1 = weight_variable([fc1_dim])
        h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

    # FC2
    with tf.name_scope("FC2"):
        fc2_dim = 128
        W_fc2 = weight_variable([fc1_dim, fc2_dim])
        b_fc2 = weight_variable([fc2_dim])
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    # FC3
    with tf.name_scope("FC3"):
        fc3_dim = 64
        W_fc3 = weight_variable([fc2_dim, fc3_dim])
        b_fc3 = weight_variable([fc3_dim])
        h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)

    # dropout
    with tf.name_scope("Dropout"):
        h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)

    # FC4
    with tf.name_scope("FC4"):
        fc4_dim = 128 
        W_fc4 = weight_variable([fc3_dim, fc4_dim])
        b_fc4 = weight_variable([fc4_dim])
        h_fc4 = tf.nn.relu(tf.matmul(h_fc3_drop, W_fc4) + b_fc4)

    # FC5
    with tf.name_scope("FC5"):
        fc5_dim = 256 
        W_fc5 = weight_variable([fc4_dim, fc5_dim])
        b_fc5 = weight_variable([fc5_dim])
        h_fc5 = tf.nn.relu(tf.matmul(h_fc4, W_fc5) + b_fc5)

    # FC6
    with tf.name_scope("FC6"):
        fc6_dim = x_dim
        W_fc6 = weight_variable([fc5_dim, fc6_dim])
        b_fc6 = weight_variable([fc6_dim])
        y = tf.matmul(h_fc5, W_fc6) + b_fc6

    # LOSS 
    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.squared_difference(y, x))
        #l1_loss = tf.reduce_sum(tf.abs(W_fc1)) + \
        #          tf.reduce_sum(tf.abs(W_fc2)) + \
        #          tf.reduce_sum(tf.abs(W_fc3)) + \
        #         tf.reduce_sum(tf.abs(W_fc4)) +  \
        #         tf.reduce_sum(tf.abs(W_fc5)) +  \
        #         tf.reduce_sum(tf.abs(W_fc6))
        #loss += gamma * l1_loss
        tf.scalar_summary("loss", loss)

        # summary
        summary_op = tf.merge_all_summaries()
    return loss, y



def dense_layer(x, dims, layer_name, activation='relu'):
    in_dim, out_dim = dims
    with tf.name_scope(layer_name):
        w = weight_variable([in_dim, out_dim])
        b = weight_variable([out_dim])
        if activation == 'relu':
            h = tf.nn.relu(tf.matmul(x, w) + b)
        elif activation == 'sigmoid':
            h = tf.nn.sigmoid(tf.matmul(x, w) + b)
        else:
            h = tf.matmul(x, w) + b
    return h


