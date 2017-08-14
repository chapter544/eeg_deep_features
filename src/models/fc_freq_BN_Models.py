
# Auto-encoder for 2D cortex slices
# Chuong Nguyen
# 07/11/2017
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from pkg_resources import parse_version
import tensorflow as tf
import numpy as np
import h5py
import glob
import os
import argparse
import sys
from datetime import datetime
from tf_utils import weight_variable, bias_variable
import tensorflow as tf

# NOTE
# With this implementation, the BN takes place before the activation function

def batch_norm_contrib(x, is_training):
    h2 = tf.contrib.layers.batch_norm(x, 
            center=True, scale=True, 
            is_training=is_training, scope='bn')

def build_fc_freq_5_NoTiedWeight_L1_BN_Tiny (
        x, x_dim, 
        keep_prob, 
        is_training, 
        gamma=1e-7, 
        activation='relu'):
    # FC1
    with tf.name_scope("FC1"):
        fc1_dim = 500
        W_fc1 = weight_variable([x_dim, fc1_dim])
        b_fc1 = weight_variable([fc1_dim])
        h_fc1_bn = tf.nn.elu(batch_norm_contrib(
                    tf.matmul(x, W_fc1) + b_fc1,
                    is_training))

    # dropout
    with tf.name_scope("Dropout1"):
        h_fc1_drop = tf.nn.dropout(h_fc1_bn, keep_prob)

    # FC2
    with tf.name_scope("FC2"):
        #fc2_dim = 4096
        fc2_dim = 300
        W_fc2 = weight_variable([fc1_dim, fc2_dim])
        b_fc2 = weight_variable([fc2_dim])
        h_fc2_bn = tf.nn.elu(batch_norm_contrib(
                    tf.matmul(h_fc1_drop, W_fc2) + b_fc2,
                    is_training))
    # dropout
    with tf.name_scope("Dropout2"):
        h_fc2_drop = tf.nn.dropout(h_fc2_bn, keep_prob)

    # FC3
    with tf.name_scope("FC3"):
        fc3_dim = 100
        W_fc3 = weight_variable([fc2_dim, fc3_dim])
        b_fc3 = weight_variable([fc3_dim])
        h_fc3_bn = tf.nn.elu(batch_norm_contrib(
                    tf.matmul(h_fc2_bn_drop, W_fc3) + b_fc3,
                    is_training))

    # FC4
    with tf.name_scope("FCFeat"):
        fc4_dim = 64
        W_fc4 = weight_variable([fc3_dim, fc4_dim])
        b_fc4 = weight_variable([fc4_dim])

        h_fc4_bn = tf.nn.elu(batch_norm_contrib(
                    tf.matmul(h_fc3_bn, W_fc4) + b_fc4,
                    is_training), name="feature")
    # FC5
    with tf.name_scope("FC5"):
        fc5_dim = 100
        W_fc5 = weight_variable([fc4_dim, fc5_dim])
        b_fc5 = weight_variable([fc5_dim])
        h_fc5_bn = tf.nn.elu(batch_norm_contrib(
                    tf.matmul(h_fc4_bn, W_fc5) + b_fc5,
                    is_training))

    # dropout
    with tf.name_scope("Dropout5"):
        h_fc5_drop = tf.nn.dropout(h_fc5_bn, keep_prob)


    with tf.name_scope("FC6"):
        fc6_dim = 300
        W_fc6 = weight_variable([fc5_dim, fc6_dim])
        b_fc6 = weight_variable([fc6_dim])
        h_fc6_bn = tf.nn.elu(batch_norm_contrib(
                    tf.matmul(h_fc5_drop, W_fc6) + b_fc6,
                    is_training))

    # dropout
    with tf.name_scope("Dropout6"):
        h_fc6_drop = tf.nn.dropout(h_fc6_bn, keep_prob)

    with tf.name_scope("FC7"):
        fc7_dim = 500
        W_fc7 = weight_variable([fc6_dim, fc7_dim])
        b_fc7 = weight_variable([fc7_dim])
        h_fc7_bn = tf.nn.elu(batch_norm_contrib(
                    tf.matmul(h_fc6_drop, W_fc7) + b_fc7,
                    is_training))

    # dropout
    with tf.name_scope("Dropout7"):
        h_fc7_drop = tf.nn.dropout(h_fc7_bn, keep_prob)

    # FC8
    with tf.name_scope("FC8"):
        fc8_dim = x_dim
        W_fc8 = weight_variable([fc7_dim, fc8_dim])
        b_fc8 = weight_variable([fc8_dim])
        h_fc8 = tf.matmul(h_fc7_drop, W_fc8) + b_fc8

    # LOSS 
    with tf.name_scope("loss"):
        y = h_fc8 + 1e-10
        l2_loss = tf.sqrt(tf.reduce_mean(tf.square(y-x)))
        #entropy_loss = - tf.reduce_mean(x * tf.log(y))

        #loss = entropy_loss + l2_loss
        #loss = l2_loss 

        l1_loss_sum = tf.reduce_sum(tf.abs(W_fc1)) + \
                      tf.reduce_sum(tf.abs(W_fc2)) + \
                      tf.reduce_sum(tf.abs(W_fc3)) + \
                      tf.reduce_sum(tf.abs(W_fc4)) + \
                      tf.reduce_sum(tf.abs(W_fc5)) +  \
                      tf.reduce_sum(tf.abs(W_fc6)) + \
                      tf.reduce_sum(tf.abs(W_fc7)) + \
                      tf.reduce_sum(tf.abs(W_fc7)) + \
                      tf.reduce_sum(tf.abs(W_fc8))
        l1_loss = l1_loss_sum * gamma
        loss = l2_loss + l1_loss
        #loss = l2_loss

        tf.summary.scalar("loss", loss)
        # summary
        summary_op = tf.summary.merge_all()

    return loss, y, l1_loss




def build_fc_freq_5_NoTiedWeight_L1_BN_Small(
        x, x_dim, 
        keep_prob, 
        is_training, 
        gamma=1e-7, 
        activation='relu'):
    # FC1
    with tf.name_scope("FC1"):
        fc1_dim = 1500
        W_fc1 = weight_variable([x_dim, fc1_dim])
        b_fc1 = weight_variable([fc1_dim])
        h_fc1_bn = tf.nn.elu(batch_norm_contrib(
                    tf.matmul(x, W_fc1) + b_fc1,
                    is_training))

    # dropout
    with tf.name_scope("Dropout1"):
        h_fc1_drop = tf.nn.dropout(h_fc1_bn, keep_prob)

    # FC2
    with tf.name_scope("FC2"):
        #fc2_dim = 4096
        fc2_dim = 1000
        W_fc2 = weight_variable([fc1_dim, fc2_dim])
        b_fc2 = weight_variable([fc2_dim])
        h_fc2_bn = tf.nn.elu(batch_norm_contrib(
                    tf.matmul(h_fc1_drop, W_fc2) + b_fc2,
                    is_training))
    # dropout
    #with tf.name_scope("Dropout2"):
    #    h_fc2_drop = tf.nn.dropout(h_fc2_bn, keep_prob)

    # FC3
    with tf.name_scope("FC3"):
        fc3_dim = 500
        W_fc3 = weight_variable([fc2_dim, fc3_dim])
        b_fc3 = weight_variable([fc3_dim])
        h_fc3_bn = tf.nn.elu(batch_norm_contrib(
                    tf.matmul(h_fc2_bn, W_fc3) + b_fc3,
                    is_training))

    # FC4
    with tf.name_scope("FCFeat"):
        fc4_dim = 64
        W_fc4 = weight_variable([fc3_dim, fc4_dim])
        b_fc4 = weight_variable([fc4_dim])

        h_fc4_bn = tf.nn.elu(batch_norm_contrib(
                    tf.matmul(h_fc3_bn, W_fc4) + b_fc4,
                    is_training), name="feature")
    # FC5
    with tf.name_scope("FC5"):
        fc5_dim = 500
        W_fc5 = weight_variable([fc4_dim, fc5_dim])
        b_fc5 = weight_variable([fc5_dim])
        h_fc5_bn = tf.nn.elu(batch_norm_contrib(
                    tf.matmul(h_fc4_bn, W_fc5) + b_fc5,
                    is_training))

    with tf.name_scope("FC6"):
        fc6_dim = 1000
        W_fc6 = weight_variable([fc5_dim, fc6_dim])
        b_fc6 = weight_variable([fc6_dim])
        h_fc6_bn = tf.nn.elu(batch_norm_contrib(
                    tf.matmul(h_fc5_bn, W_fc6) + b_fc6,
                    is_training))

    # dropout
    with tf.name_scope("Dropout6"):
        h_fc6_drop = tf.nn.dropout(h_fc6_bn, keep_prob)

    with tf.name_scope("FC7"):
        fc7_dim = 1500
        W_fc7 = weight_variable([fc6_dim, fc7_dim])
        b_fc7 = weight_variable([fc7_dim])
        h_fc7_bn = tf.nn.elu(batch_norm_contrib(
                    tf.matmul(h_fc6_drop, W_fc7) + b_fc7,
                    is_training))

    # dropout
    with tf.name_scope("Dropout7"):
        h_fc7_drop = tf.nn.dropout(h_fc7_bn, keep_prob)

    # FC8
    with tf.name_scope("FC8"):
        fc8_dim = x_dim
        W_fc8 = weight_variable([fc7_dim, fc8_dim])
        b_fc8 = weight_variable([fc8_dim])
        h_fc8 = tf.matmul(h_fc7_drop, W_fc8) + b_fc8

    # LOSS 
    with tf.name_scope("loss"):
        y = h_fc8 + 1e-10
        l2_loss = tf.sqrt(tf.reduce_mean(tf.square(y-x)))
        #entropy_loss = - tf.reduce_mean(x * tf.log(y))

        #loss = entropy_loss + l2_loss
        #loss = l2_loss 

        l1_loss_sum = tf.reduce_sum(tf.abs(W_fc1)) + \
                      tf.reduce_sum(tf.abs(W_fc2)) + \
                      tf.reduce_sum(tf.abs(W_fc3)) + \
                      tf.reduce_sum(tf.abs(W_fc4)) + \
                      tf.reduce_sum(tf.abs(W_fc5)) +  \
                      tf.reduce_sum(tf.abs(W_fc6)) + \
                      tf.reduce_sum(tf.abs(W_fc7)) + \
                      tf.reduce_sum(tf.abs(W_fc7)) + \
                      tf.reduce_sum(tf.abs(W_fc8))
        l1_loss = l1_loss_sum * gamma
        #loss = l2_loss + l1_loss
        loss = l2_loss

        tf.summary.scalar("loss", loss)
        # summary
        summary_op = tf.summary.merge_all()

    return loss, y, l1_loss




def build_fc_freq_5_NoTiedWeight_Small(
        x, x_dim, 
        keep_prob, 
        is_training, 
        gamma=1e-7, 
        activation='relu'):
    # FC1
    with tf.name_scope("FC1"):
        fc1_dim = 1500
        W_fc1 = weight_variable([x_dim, fc1_dim])
        b_fc1 = weight_variable([fc1_dim])
        h_fc1 = tf.nn.elu(tf.matmul(x, W_fc1) + b_fc1)
        h_fc1_bn = batch_norm_contrib(h_fc1, is_training)

    # dropout
    #with tf.name_scope("Dropout1"):
    #    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # FC2
    with tf.name_scope("FC2"):
        #fc2_dim = 4096
        fc2_dim = 1000
        W_fc2 = weight_variable([fc1_dim, fc2_dim])
        b_fc2 = weight_variable([fc2_dim])
        h_fc2 = tf.nn.elu(tf.matmul(h_fc1_bn, W_fc2) + b_fc2)
        h_fc2_bn = batch_norm_contrib(h_fc2, is_training)

    # dropout
    #with tf.name_scope("Dropout2"):
    #    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    # FC3
    with tf.name_scope("FC3"):
        fc3_dim = 500
        W_fc3 = weight_variable([fc2_dim, fc3_dim])
        b_fc3 = weight_variable([fc3_dim])
        h_fc3 = tf.nn.elu(tf.matmul(h_fc2_bn, W_fc3) + b_fc3)
        h_fc3_bn = batch_norm_contrib(h_fc3, is_training)

    # FC4
    with tf.name_scope("FCFeat"):
        fc4_dim = 64
        W_fc4 = weight_variable([fc3_dim, fc4_dim])
        b_fc4 = weight_variable([fc4_dim])
        h_fc4 = tf.nn.elu(tf.matmul(h_fc3_bn, W_fc4) + b_fc4, name="feature")
        h_fc4_bn = batch_norm_contrib(h_fc4, is_training)

    # FC5
    with tf.name_scope("FC5"):
        fc5_dim = 500
        W_fc5 = weight_variable([fc4_dim, fc5_dim])
        b_fc5 = weight_variable([fc5_dim])
        h_fc5 = tf.nn.elu(tf.matmul(h_fc4_bn, W_fc5) + b_fc5)
        h_fc5_bn = batch_norm_contrib(h_fc5, is_training)

    with tf.name_scope("FC6"):
        fc6_dim = 1000
        W_fc6 = weight_variable([fc5_dim, fc6_dim])
        b_fc6 = weight_variable([fc6_dim])
        h_fc6 = tf.nn.elu(tf.matmul(h_fc5_bn, W_fc6) + b_fc6)
        h_fc6_bn = batch_norm_contrib(h_fc6, is_training)


    with tf.name_scope("FC7"):
        fc7_dim = 1500
        W_fc7 = weight_variable([fc6_dim, fc7_dim])
        b_fc7 = weight_variable([fc7_dim])
        h_fc7 = tf.nn.elu(tf.matmul(h_fc6_bn, W_fc7) + b_fc7)
        h_fc7_bn = batch_norm_contrib(h_fc7, is_training)

    # dropout
    #with tf.name_scope("Dropout7"):
    #    h_fc7_drop = tf.nn.dropout(h_fc7, keep_prob)

    # FC8
    with tf.name_scope("FC8"):
        fc8_dim = x_dim
        W_fc8 = weight_variable([fc7_dim, fc8_dim])
        b_fc8 = weight_variable([fc8_dim])
        h_fc8 = tf.matmul(h_fc7_bn, W_fc8) + b_fc8

    # LOSS 
    with tf.name_scope("loss"):
        y = h_fc8 + 1e-10
        l2_loss = tf.sqrt(tf.reduce_mean(tf.square(y-x)))
        #entropy_loss = - tf.reduce_mean(x * tf.log(y))

        #loss = entropy_loss + l2_loss
        loss = l2_loss 

        l1_loss_sum = tf.reduce_sum(tf.abs(W_fc1)) + \
                      tf.reduce_sum(tf.abs(W_fc2)) + \
                      tf.reduce_sum(tf.abs(W_fc3)) + \
                      tf.reduce_sum(tf.abs(W_fc4)) + \
                      tf.reduce_sum(tf.abs(W_fc5)) +  \
                      tf.reduce_sum(tf.abs(W_fc6)) + \
                      tf.reduce_sum(tf.abs(W_fc7)) + \
                      tf.reduce_sum(tf.abs(W_fc7))
        l1_loss = l1_loss_sum * gamma

        tf_version = tf.__version__.rpartition('.')[0]
        if parse_version(tf_version) >= parse_version('0.12.0'):
            tf.summary.scalar("loss", loss)
        else:
            tf.scalar_summary("loss", loss)

        # summary
        if parse_version(tf_version) >= parse_version('0.12.0'):
            summary_op = tf.summary.merge_all()
        else:
            summary_op = tf.merge_all_summaries()

    return loss, y, l1_loss




def build_fc_freq_5_NoTiedWeight_Big(
        x, x_dim, 
        keep_prob, 
        is_training, 
        gamma=1e-7, 
        activation='relu'):
    # FC1
    with tf.name_scope("EN1"):
        fc1_dim = 1200
        W_fc1 = weight_variable([x_dim, fc1_dim])
        b_fc1 = weight_variable([fc1_dim])
        h_fc1 = tf.nn.elu(tf.matmul(x, W_fc1) + b_fc1)
        h_fc1_bn = batch_norm_contrib(h_fc1, is_training)

    # dropout
    #with tf.name_scope("Dropout1"):
    #    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # FC2
    with tf.name_scope("EN2"):
        #fc2_dim = 4096
        fc2_dim = 1000
        W_fc2 = weight_variable([fc1_dim, fc2_dim])
        b_fc2 = weight_variable([fc2_dim])
        h_fc2 = tf.nn.elu(tf.matmul(h_fc1_bn, W_fc2) + b_fc2)
        h_fc2_bn = batch_norm_contrib(h_fc2, is_training)

    # dropout
    #with tf.name_scope("Dropout2"):
    #    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    # FC3
    with tf.name_scope("EN3"):
        fc3_dim = 800 
        W_fc3 = weight_variable([fc2_dim, fc3_dim])
        b_fc3 = weight_variable([fc3_dim])
        h_fc3 = tf.nn.elu(tf.matmul(h_fc2_bn, W_fc3) + b_fc3)
        h_fc3_bn = batch_norm_contrib(h_fc3, is_training)


    # FC3
    with tf.name_scope("EN4"):
        fc4_dim = 500
        W_fc4 = weight_variable([fc3_dim, fc4_dim])
        b_fc4 = weight_variable([fc4_dim])
        h_fc4 = tf.nn.elu(tf.matmul(h_fc3_bn, W_fc4) + b_fc4)
        h_fc4_bn = batch_norm_contrib(h_fc4, is_training)


    # FC3
    with tf.name_scope("EN5"):
        fc41_dim = 200
        W_fc41 = weight_variable([fc4_dim, fc41_dim])
        b_fc41 = weight_variable([fc41_dim])
        h_fc41 = tf.nn.elu(tf.matmul(h_fc4_bn, W_fc41) + b_fc41)
        h_fc41_bn = batch_norm_contrib(h_fc41, is_training)


    # FC4
    with tf.name_scope("FCFeat"):
        fc_dim = 64
        W_fc = weight_variable([fc41_dim, fc_dim])
        b_fc = weight_variable([fc_dim])
        h_fc = tf.nn.elu(tf.matmul(h_fc41_bn, W_fc) + b_fc, name="feature")
        h_fc_bn = batch_norm_contrib(h_fc, is_training)

    # FC5
    with tf.name_scope("DE1"):
        fc5_dim = 200
        #W_fc5 = tf.transpose(W_fc)
        W_fc5 = weight_variable([fc_dim, fc5_dim])
        b_fc5 = weight_variable([fc5_dim])
        h_fc5 = tf.nn.elu(tf.matmul(h_fc_bn, W_fc5) + b_fc5)
        h_fc5_bn = batch_norm_contrib(h_fc5, is_training)

    with tf.name_scope("DE2"):
        fc6_dim = 500
        #W_fc6 = tf.transpose(W_fc41)
        W_fc6 = weight_variable([fc5_dim, fc6_dim])
        b_fc6 = weight_variable([fc6_dim])
        h_fc6 = tf.nn.elu(tf.matmul(h_fc5_bn, W_fc6) + b_fc6)
        h_fc6_bn = batch_norm_contrib(h_fc6, is_training)


    with tf.name_scope("DE3"):
        fc7_dim = 800
        #W_fc7 = tf.transpose(W_fc4)
        W_fc7 = weight_variable([fc6_dim, fc7_dim])
        b_fc7 = weight_variable([fc7_dim])
        h_fc7 = tf.nn.elu(tf.matmul(h_fc6_bn, W_fc7) + b_fc7)
        h_fc7_bn = batch_norm_contrib(h_fc7, is_training)

    with tf.name_scope("DE4"):
        fc8_dim = 1000
        #W_fc8 = tf.transpose(W_fc3)
        W_fc8 = weight_variable([fc7_dim, fc8_dim])
        b_fc8 = weight_variable([fc8_dim])
        h_fc8 = tf.nn.elu(tf.matmul(h_fc7_bn, W_fc8) + b_fc8)
        h_fc8_bn = batch_norm_contrib(h_fc8, is_training)

    with tf.name_scope("DE5"):
        fc81_dim = 1200
        #W_fc81 = tf.transpose(W_fc2)
        W_fc81 = weight_variable([fc8_dim, fc81_dim])
        b_fc81 = weight_variable([fc81_dim])
        h_fc81 = tf.nn.elu(tf.matmul(h_fc8_bn, W_fc81) + b_fc81)
        h_fc81_bn = batch_norm_contrib(h_fc81, is_training)


    # dropout
    #with tf.name_scope("Dropout7"):
    #    h_fc7_drop = tf.nn.dropout(h_fc7, keep_prob)

    # FC8
    with tf.name_scope("OUT"):
        fc9_dim = x_dim
        W_fc9 = weight_variable([fc81_dim, fc9_dim])
        #W_fc9 = tf.transpose(W_fc1)
        b_fc9 = weight_variable([fc9_dim])
        h_fc9 = tf.matmul(h_fc81_bn, W_fc9) + b_fc9

    # LOSS 
    with tf.name_scope("loss"):
        y = h_fc9 + 1e-10
        l2_loss = tf.sqrt(tf.reduce_mean(tf.square(y-x)))
        #entropy_loss = - tf.reduce_mean(x * tf.log(y))

        #loss = entropy_loss + l2_loss
        loss = l2_loss 

        l1_loss_sum = tf.reduce_sum(tf.abs(W_fc1)) + \
                      tf.reduce_sum(tf.abs(W_fc2)) + \
                      tf.reduce_sum(tf.abs(W_fc3)) + \
                      tf.reduce_sum(tf.abs(W_fc4)) + \
                      tf.reduce_sum(tf.abs(W_fc41)) +  \
                      tf.reduce_sum(tf.abs(W_fc))
        l1_loss = l1_loss_sum * gamma
        #loss += l1_loss

        tf_version = tf.__version__.rpartition('.')[0]
        if parse_version(tf_version) >= parse_version('0.12.0'):
            tf.summary.scalar("loss", loss)
        else:
            tf.scalar_summary("loss", loss)

        # summary
        if parse_version(tf_version) >= parse_version('0.12.0'):
            summary_op = tf.summary.merge_all()
        else:
            summary_op = tf.merge_all_summaries()

    return loss, y, l1_loss




def build_fc_freq_5_TiedWeight_Big(
        x, x_dim, 
        keep_prob, 
        is_training, 
        gamma=1e-7, 
        activation='relu'):
    # FC1
    with tf.name_scope("EN1"):
        fc1_dim = 2000
        W_fc1 = weight_variable([x_dim, fc1_dim])
        b_fc1 = weight_variable([fc1_dim])
        h_fc1 = tf.nn.elu(tf.matmul(x, W_fc1) + b_fc1)
        h_fc1_bn = batch_norm_contrib(h_fc1, is_training)

    # dropout
    #with tf.name_scope("Dropout1"):
    #    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # FC2
    with tf.name_scope("EN2"):
        #fc2_dim = 4096
        fc2_dim = 1500
        W_fc2 = weight_variable([fc1_dim, fc2_dim])
        b_fc2 = weight_variable([fc2_dim])
        h_fc2 = tf.nn.elu(tf.matmul(h_fc1_bn, W_fc2) + b_fc2)
        h_fc2_bn = batch_norm_contrib(h_fc2, is_training)

    # dropout
    #with tf.name_scope("Dropout2"):
    #    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    # FC3
    with tf.name_scope("EN3"):
        fc3_dim = 1000 
        W_fc3 = weight_variable([fc2_dim, fc3_dim])
        b_fc3 = weight_variable([fc3_dim])
        h_fc3 = tf.nn.elu(tf.matmul(h_fc2_bn, W_fc3) + b_fc3)
        h_fc3_bn = batch_norm_contrib(h_fc3, is_training)


    # FC3
    with tf.name_scope("EN4"):
        fc4_dim = 500
        W_fc4 = weight_variable([fc3_dim, fc4_dim])
        b_fc4 = weight_variable([fc4_dim])
        h_fc4 = tf.nn.elu(tf.matmul(h_fc3_bn, W_fc4) + b_fc4)
        h_fc4_bn = batch_norm_contrib(h_fc4, is_training)


    # FC3
    with tf.name_scope("EN5"):
        fc41_dim = 200
        W_fc41 = weight_variable([fc4_dim, fc41_dim])
        b_fc41 = weight_variable([fc41_dim])
        h_fc41 = tf.nn.elu(tf.matmul(h_fc4_bn, W_fc41) + b_fc41)
        h_fc41_bn = batch_norm_contrib(h_fc41, is_training)


    # FC4
    with tf.name_scope("FCFeat"):
        fc_dim = 64
        W_fc = weight_variable([fc41_dim, fc_dim])
        b_fc = weight_variable([fc_dim])
        h_fc = tf.nn.elu(tf.matmul(h_fc41_bn, W_fc) + b_fc, name="feature")
        h_fc_bn = batch_norm_contrib(h_fc, is_training)

    # FC5
    with tf.name_scope("DE1"):
        fc5_dim = 200
        W_fc5 = tf.transpose(W_fc)
        b_fc5 = weight_variable([fc5_dim])
        h_fc5 = tf.nn.elu(tf.matmul(h_fc_bn, W_fc5) + b_fc5)
        h_fc5_bn = batch_norm_contrib(h_fc5, is_training)

    with tf.name_scope("DE2"):
        fc6_dim = 500
        W_fc6 = tf.transpose(W_fc41)
        b_fc6 = weight_variable([fc6_dim])
        h_fc6 = tf.nn.elu(tf.matmul(h_fc5_bn, W_fc6) + b_fc6)
        h_fc6_bn = batch_norm_contrib(h_fc6, is_training)


    with tf.name_scope("DE3"):
        fc7_dim = 1000
        W_fc7 = tf.transpose(W_fc4)
        b_fc7 = weight_variable([fc7_dim])
        h_fc7 = tf.nn.elu(tf.matmul(h_fc6_bn, W_fc7) + b_fc7)
        h_fc7_bn = batch_norm_contrib(h_fc7, is_training)

    with tf.name_scope("DE4"):
        fc8_dim = 1500
        W_fc8 = tf.transpose(W_fc3)
        b_fc8 = weight_variable([fc8_dim])
        h_fc8 = tf.nn.elu(tf.matmul(h_fc7_bn, W_fc8) + b_fc8)
        h_fc8_bn = batch_norm_contrib(h_fc8, is_training)

    with tf.name_scope("DE5"):
        fc81_dim = 2000
        W_fc81 = tf.transpose(W_fc2)
        b_fc81 = weight_variable([fc81_dim])
        h_fc81 = tf.nn.elu(tf.matmul(h_fc8_bn, W_fc81) + b_fc81)
        h_fc81_bn = batch_norm_contrib(h_fc81, is_training)


    # dropout
    #with tf.name_scope("Dropout7"):
    #    h_fc7_drop = tf.nn.dropout(h_fc7, keep_prob)

    # FC8
    with tf.name_scope("OUT"):
        fc9_dim = x_dim
        #W_fc8 = weight_variable([fc7_dim, fc8_dim])
        W_fc9 = tf.transpose(W_fc1)
        b_fc9 = weight_variable([fc9_dim])
        h_fc9 = tf.matmul(h_fc81_bn, W_fc9) + b_fc9

    # LOSS 
    with tf.name_scope("loss"):
        y = h_fc9 + 1e-10
        l2_loss = tf.sqrt(tf.reduce_mean(tf.square(y-x)))
        #entropy_loss = - tf.reduce_mean(x * tf.log(y))

        #loss = entropy_loss + l2_loss
        loss = l2_loss 

        l1_loss_sum = tf.reduce_sum(tf.abs(W_fc1)) + \
                      tf.reduce_sum(tf.abs(W_fc2)) + \
                      tf.reduce_sum(tf.abs(W_fc3)) + \
                      tf.reduce_sum(tf.abs(W_fc4)) + \
                      tf.reduce_sum(tf.abs(W_fc41)) +  \
                      tf.reduce_sum(tf.abs(W_fc))
        l1_loss = l1_loss_sum * gamma
        #loss += l1_loss

        tf_version = tf.__version__.rpartition('.')[0]
        if parse_version(tf_version) >= parse_version('0.12.0'):
            tf.summary.scalar("loss", loss)
        else:
            tf.scalar_summary("loss", loss)

        # summary
        if parse_version(tf_version) >= parse_version('0.12.0'):
            summary_op = tf.summary.merge_all()
        else:
            summary_op = tf.merge_all_summaries()

    return loss, y, l1_loss



def build_fc_freq_5_TiedWeight_L1_Tiny(
        x, x_dim, 
        keep_prob, 
        is_training, 
        gamma=1e-7, 
        activation='relu'):
    # FC1
    with tf.name_scope("FC1"):
        fc1_dim = 800
        W_fc1 = weight_variable([x_dim, fc1_dim])
        b_fc1 = weight_variable([fc1_dim])
        h_fc1 = tf.nn.elu(tf.matmul(x, W_fc1) + b_fc1)
        h_fc1_bn = batch_norm_contrib(h_fc1, is_training)

    # dropout
    #with tf.name_scope("Dropout1"):
    #    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # FC2
    with tf.name_scope("FC2"):
        #fc2_dim = 4096
        fc2_dim = 500
        W_fc2 = weight_variable([fc1_dim, fc2_dim])
        b_fc2 = weight_variable([fc2_dim])
        h_fc2 = tf.nn.elu(tf.matmul(h_fc1_bn, W_fc2) + b_fc2)
        h_fc2_bn = batch_norm_contrib(h_fc2, is_training)

    # dropout
    #with tf.name_scope("Dropout2"):
    #    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    # FC3
    with tf.name_scope("FC3"):
        fc3_dim = 200
        W_fc3 = weight_variable([fc2_dim, fc3_dim])
        b_fc3 = weight_variable([fc3_dim])
        h_fc3 = tf.nn.elu(tf.matmul(h_fc2_bn, W_fc3) + b_fc3)
        h_fc3_bn = batch_norm_contrib(h_fc3, is_training)

    # FC4
    with tf.name_scope("FCFeat"):
        fc4_dim = 64
        W_fc4 = weight_variable([fc3_dim, fc4_dim])
        b_fc4 = weight_variable([fc4_dim])
        h_fc4 = tf.nn.elu(tf.matmul(h_fc3_bn, W_fc4) + b_fc4, name="feature")
        h_fc4_bn = batch_norm_contrib(h_fc4, is_training)

    # FC5
    with tf.name_scope("FC5"):
        fc5_dim = 200
        W_fc5 = tf.transpose(W_fc4)
        b_fc5 = weight_variable([fc5_dim])
        h_fc5 = tf.nn.elu(tf.matmul(h_fc4_bn, W_fc5) + b_fc5)
        h_fc5_bn = batch_norm_contrib(h_fc5, is_training)

    with tf.name_scope("FC6"):
        fc6_dim = 500
        W_fc6 = tf.transpose(W_fc3)
        b_fc6 = weight_variable([fc6_dim])
        h_fc6 = tf.nn.elu(tf.matmul(h_fc5_bn, W_fc6) + b_fc6)
        h_fc6_bn = batch_norm_contrib(h_fc6, is_training)


    with tf.name_scope("FC7"):
        fc7_dim = 800
        W_fc7 = tf.transpose(W_fc2)
        b_fc7 = weight_variable([fc7_dim])
        h_fc7 = tf.nn.elu(tf.matmul(h_fc6_bn, W_fc7) + b_fc7)
        h_fc7_bn = batch_norm_contrib(h_fc7, is_training)

    # dropout
    #with tf.name_scope("Dropout7"):
    #    h_fc7_drop = tf.nn.dropout(h_fc7, keep_prob)

    # FC8
    with tf.name_scope("FC8"):
        fc8_dim = x_dim
        #W_fc8 = weight_variable([fc7_dim, fc8_dim])
        W_fc8 = tf.transpose(W_fc1)
        b_fc8 = weight_variable([fc8_dim])
        h_fc8 = tf.matmul(h_fc7_bn, W_fc8) + b_fc8

    # LOSS 
    with tf.name_scope("loss"):
        y = h_fc8 + 1e-10
        l2_loss = tf.sqrt(tf.reduce_mean(tf.square(y-x)))
        #entropy_loss = - tf.reduce_mean(x * tf.log(y))

        #loss = entropy_loss + l2_loss
        loss = l2_loss 

        l1_loss_sum = tf.reduce_sum(tf.abs(W_fc1)) + \
                      tf.reduce_sum(tf.abs(W_fc2)) + \
                      tf.reduce_sum(tf.abs(W_fc3)) + \
                      tf.reduce_sum(tf.abs(W_fc4)) + \
                      tf.reduce_sum(tf.abs(W_fc5)) + \
                      tf.reduce_sum(tf.abs(W_fc6)) + \
                      tf.reduce_sum(tf.abs(W_fc7)) + \
                      tf.reduce_sum(tf.abs(W_fc8))
        l1_loss = l1_loss_sum * gamma
        loss += l1_loss

        tf.summary.scalar("loss", loss)
        summary_op = tf.summary.merge_all()

    return loss, y, l1_loss




def build_fc_freq_5_TiedWeight_L1_Small(
        x, x_dim, 
        keep_prob, 
        is_training, 
        gamma=1e-7, 
        activation='relu'):
    # FC1
    with tf.name_scope("FC1"):
        fc1_dim = 1024
        W_fc1 = weight_variable([x_dim, fc1_dim])
        b_fc1 = weight_variable([fc1_dim])
        h_fc1 = tf.nn.elu(tf.matmul(x, W_fc1) + b_fc1)
        h_fc1_bn = batch_norm_contrib(h_fc1, is_training)

    # dropout
    #with tf.name_scope("Dropout1"):
    #    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # FC2
    with tf.name_scope("FC2"):
        #fc2_dim = 4096
        fc2_dim = 512
        W_fc2 = weight_variable([fc1_dim, fc2_dim])
        b_fc2 = weight_variable([fc2_dim])
        h_fc2 = tf.nn.elu(tf.matmul(h_fc1_bn, W_fc2) + b_fc2)
        h_fc2_bn = batch_norm_contrib(h_fc2, is_training)

    # dropout
    #with tf.name_scope("Dropout2"):
    #    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    # FC3
    with tf.name_scope("FC3"):
        fc3_dim = 256
        W_fc3 = weight_variable([fc2_dim, fc3_dim])
        b_fc3 = weight_variable([fc3_dim])
        h_fc3 = tf.nn.elu(tf.matmul(h_fc2_bn, W_fc3) + b_fc3)
        h_fc3_bn = batch_norm_contrib(h_fc3, is_training)

    # FC4
    with tf.name_scope("FCFeat"):
        fc4_dim = 64
        W_fc4 = weight_variable([fc3_dim, fc4_dim])
        b_fc4 = weight_variable([fc4_dim])
        h_fc4 = tf.nn.elu(tf.matmul(h_fc3_bn, W_fc4) + b_fc4, name="feature")
        h_fc4_bn = batch_norm_contrib(h_fc4, is_training)

    # FC5
    with tf.name_scope("FC5"):
        fc5_dim = 256
        W_fc5 = tf.transpose(W_fc4)
        b_fc5 = weight_variable([fc5_dim])
        h_fc5 = tf.nn.elu(tf.matmul(h_fc4_bn, W_fc5) + b_fc5)
        h_fc5_bn = batch_norm_contrib(h_fc5, is_training)

    with tf.name_scope("FC6"):
        fc6_dim = 512
        W_fc6 = tf.transpose(W_fc3)
        b_fc6 = weight_variable([fc6_dim])
        h_fc6 = tf.nn.elu(tf.matmul(h_fc5_bn, W_fc6) + b_fc6)
        h_fc6_bn = batch_norm_contrib(h_fc6, is_training)


    with tf.name_scope("FC7"):
        fc7_dim = 1024
        W_fc7 = tf.transpose(W_fc2)
        b_fc7 = weight_variable([fc7_dim])
        h_fc7 = tf.nn.elu(tf.matmul(h_fc6_bn, W_fc7) + b_fc7)
        h_fc7_bn = batch_norm_contrib(h_fc7, is_training)

    # dropout
    #with tf.name_scope("Dropout7"):
    #    h_fc7_drop = tf.nn.dropout(h_fc7, keep_prob)

    # FC8
    with tf.name_scope("FC8"):
        fc8_dim = x_dim
        #W_fc8 = weight_variable([fc7_dim, fc8_dim])
        W_fc8 = tf.transpose(W_fc1)
        b_fc8 = weight_variable([fc8_dim])
        h_fc8 = tf.matmul(h_fc7_bn, W_fc8) + b_fc8

    # LOSS 
    with tf.name_scope("loss"):
        y = h_fc8 + 1e-10
        l2_loss = tf.sqrt(tf.reduce_mean(tf.square(y-x)))
        #entropy_loss = - tf.reduce_mean(x * tf.log(y))

        #loss = entropy_loss + l2_loss
        loss = l2_loss 

        l1_loss_sum = tf.reduce_sum(tf.abs(W_fc1)) + \
                      tf.reduce_sum(tf.abs(W_fc2)) + \
                      tf.reduce_sum(tf.abs(W_fc3)) + \
                      tf.reduce_sum(tf.abs(W_fc4)) + \
                      tf.reduce_sum(tf.abs(W_fc5)) + \
                      tf.reduce_sum(tf.abs(W_fc6)) + \
                      tf.reduce_sum(tf.abs(W_fc7)) + \
                      tf.reduce_sum(tf.abs(W_fc8))
        l1_loss = l1_loss_sum * gamma
        loss += l1_loss

        tf.summary.scalar("loss", loss)
        summary_op = tf.summary.merge_all()

    return loss, y, l1_loss



def build_fc_freq_5_TiedWeight_Small(
        x, x_dim, 
        keep_prob, 
        is_training, 
        gamma=1e-7, 
        activation='relu'):
    # FC1
    with tf.name_scope("FC1"):
        fc1_dim = 1024
        W_fc1 = weight_variable([x_dim, fc1_dim])
        b_fc1 = weight_variable([fc1_dim])
        h_fc1 = tf.nn.elu(tf.matmul(x, W_fc1) + b_fc1)
        h_fc1_bn = batch_norm_contrib(h_fc1, is_training)

    # dropout
    #with tf.name_scope("Dropout1"):
    #    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # FC2
    with tf.name_scope("FC2"):
        #fc2_dim = 4096
        fc2_dim = 512
        W_fc2 = weight_variable([fc1_dim, fc2_dim])
        b_fc2 = weight_variable([fc2_dim])
        h_fc2 = tf.nn.elu(tf.matmul(h_fc1_bn, W_fc2) + b_fc2)
        h_fc2_bn = batch_norm_contrib(h_fc2, is_training)

    # dropout
    #with tf.name_scope("Dropout2"):
    #    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    # FC3
    with tf.name_scope("FC3"):
        fc3_dim = 256
        W_fc3 = weight_variable([fc2_dim, fc3_dim])
        b_fc3 = weight_variable([fc3_dim])
        h_fc3 = tf.nn.elu(tf.matmul(h_fc2_bn, W_fc3) + b_fc3)
        h_fc3_bn = batch_norm_contrib(h_fc3, is_training)

    # FC4
    with tf.name_scope("FCFeat"):
        fc4_dim = 64
        W_fc4 = weight_variable([fc3_dim, fc4_dim])
        b_fc4 = weight_variable([fc4_dim])
        h_fc4 = tf.nn.elu(tf.matmul(h_fc3_bn, W_fc4) + b_fc4, name="feature")
        h_fc4_bn = batch_norm_contrib(h_fc4, is_training)

    # FC5
    with tf.name_scope("FC5"):
        fc5_dim = 256
        W_fc5 = tf.transpose(W_fc4)
        b_fc5 = weight_variable([fc5_dim])
        h_fc5 = tf.nn.elu(tf.matmul(h_fc4_bn, W_fc5) + b_fc5)
        h_fc5_bn = batch_norm_contrib(h_fc5, is_training)

    with tf.name_scope("FC6"):
        fc6_dim = 512
        W_fc6 = tf.transpose(W_fc3)
        b_fc6 = weight_variable([fc6_dim])
        h_fc6 = tf.nn.elu(tf.matmul(h_fc5_bn, W_fc6) + b_fc6)
        h_fc6_bn = batch_norm_contrib(h_fc6, is_training)


    with tf.name_scope("FC7"):
        fc7_dim = 1024
        W_fc7 = tf.transpose(W_fc2)
        b_fc7 = weight_variable([fc7_dim])
        h_fc7 = tf.nn.elu(tf.matmul(h_fc6_bn, W_fc7) + b_fc7)
        h_fc7_bn = batch_norm_contrib(h_fc7, is_training)

    # dropout
    #with tf.name_scope("Dropout7"):
    #    h_fc7_drop = tf.nn.dropout(h_fc7, keep_prob)

    # FC8
    with tf.name_scope("FC8"):
        fc8_dim = x_dim
        #W_fc8 = weight_variable([fc7_dim, fc8_dim])
        W_fc8 = tf.transpose(W_fc1)
        b_fc8 = weight_variable([fc8_dim])
        h_fc8 = tf.matmul(h_fc7_bn, W_fc8) + b_fc8

    # LOSS 
    with tf.name_scope("loss"):
        y = h_fc8 + 1e-10
        l2_loss = tf.sqrt(tf.reduce_mean(tf.square(y-x)))
        #entropy_loss = - tf.reduce_mean(x * tf.log(y))

        #loss = entropy_loss + l2_loss
        loss = l2_loss 

        l1_loss_sum = tf.reduce_sum(tf.abs(W_fc1)) + \
                      tf.reduce_sum(tf.abs(W_fc2)) + \
                      tf.reduce_sum(tf.abs(W_fc3)) + \
                      tf.reduce_sum(tf.abs(W_fc4))
        #          tf.reduce_sum(tf.abs(W_fc5)) +  \
        #          tf.reduce_sum(tf.abs(W_fc6))
        l1_loss = l1_loss_sum * gamma
        #loss += l1_loss

        tf_version = tf.__version__.rpartition('.')[0]
        if parse_version(tf_version) >= parse_version('0.12.0'):
            tf.summary.scalar("loss", loss)
        else:
            tf.scalar_summary("loss", loss)

        # summary
        if parse_version(tf_version) >= parse_version('0.12.0'):
            summary_op = tf.summary.merge_all()
        else:
            summary_op = tf.merge_all_summaries()

    return loss, y, l1_loss





def build_fc_freq_4_30_TiedWeight_Small(
        x, x_dim, 
        keep_prob, 
        is_training,
        gamma=1e-7, 
        activation='elu'):
    # FC1
    with tf.name_scope("FC1"):
        fc1_dim = 512
        W_fc1 = weight_variable([x_dim, fc1_dim])
        b_fc1 = weight_variable([fc1_dim])
        h_fc1 = tf.nn.elu(tf.matmul(x, W_fc1) + b_fc1)
        h_fc1_bn = batch_norm_contrib(h_fc1, is_training)

    # dropout
    #with tf.name_scope("Dropout1"):
    #    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # FC2
    with tf.name_scope("FC2"):
        #fc2_dim = 4096
        fc2_dim = 256
        W_fc2 = weight_variable([fc1_dim, fc2_dim])
        b_fc2 = weight_variable([fc2_dim])
        h_fc2 = tf.nn.elu(tf.matmul(h_fc1_bn, W_fc2) + b_fc2)
        h_fc2_bn = batch_norm_contrib(h_fc2, is_training)

    # dropout
    #with tf.name_scope("Dropout2"):
    #    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    # FC3
    with tf.name_scope("FC3"):
        fc3_dim = 128
        W_fc3 = weight_variable([fc2_dim, fc3_dim])
        b_fc3 = weight_variable([fc3_dim])
        h_fc3 = tf.nn.elu(tf.matmul(h_fc2_bn, W_fc3) + b_fc3)
        h_fc3_bn = batch_norm_contrib(h_fc3, is_training)

    # FC4
    with tf.name_scope("FCFeat"):
        fc4_dim = 64
        W_fc4 = weight_variable([fc3_dim, fc4_dim])
        b_fc4 = weight_variable([fc4_dim])
        h_fc4 = tf.nn.elu(tf.matmul(h_fc3, W_fc4) + b_fc4, name="feature")
        h_fc4_bn = batch_norm_contrib(h_fc4, is_training)

    # FC5
    with tf.name_scope("FC5"):
        fc5_dim = 128
        #W_fc5 = weight_variable([fc4_dim, fc5_dim])
        W_fc5 = tf.transpose(W_fc4)
        b_fc5 = weight_variable([fc5_dim])
        h_fc5 = tf.nn.elu(tf.matmul(h_fc4_bn, W_fc5) + b_fc5)
        h_fc5_bn = batch_norm_contrib(h_fc5, is_training)

    with tf.name_scope("FC6"):
        fc6_dim = 256
        W_fc6 = tf.transpose(W_fc3)
        b_fc6 = weight_variable([fc6_dim])
        h_fc6 = tf.nn.elu(tf.matmul(h_fc5_bn, W_fc6) + b_fc6)
        h_fc6_bn = batch_norm_contrib(h_fc6, is_training)


    with tf.name_scope("FC7"):
        fc7_dim = 512
        W_fc7 = tf.transpose(W_fc2)
        b_fc7 = weight_variable([fc7_dim])
        h_fc7 = tf.nn.elu(tf.matmul(h_fc6_bn, W_fc7) + b_fc7)
        h_fc7_bn = batch_norm_contrib(h_fc7, is_training)

    # dropout
    #with tf.name_scope("Dropout7"):
    #    h_fc7_drop = tf.nn.dropout(h_fc7, keep_prob)

    # FC8
    with tf.name_scope("FC8"):
        fc8_dim = x_dim
        #W_fc8 = weight_variable([fc7_dim, fc8_dim])
        W_fc8 = tf.transpose(W_fc1)
        b_fc8 = weight_variable([fc8_dim])
        h_fc8 = tf.matmul(h_fc7_bn, W_fc8) + b_fc8

    # LOSS 
    with tf.name_scope("loss"):
        y = h_fc8 + 1e-10
        l2_loss = tf.sqrt(tf.reduce_mean(tf.square(y-x)))
        #entropy_loss = - tf.reduce_mean(x * tf.log(y))

        #loss = entropy_loss + l2_loss
        loss = l2_loss 

        l1_loss_sum = tf.reduce_sum(tf.abs(W_fc1)) + \
                      tf.reduce_sum(tf.abs(W_fc2)) + \
                      tf.reduce_sum(tf.abs(W_fc3)) + \
                      tf.reduce_sum(tf.abs(W_fc4))
        #          tf.reduce_sum(tf.abs(W_fc5)) +  \
        #          tf.reduce_sum(tf.abs(W_fc6))
        l1_loss = l1_loss_sum * gamma
        #loss += l1_loss

        tf_version = tf.__version__.rpartition('.')[0]
        if parse_version(tf_version) >= parse_version('0.12.0'):
            tf.summary.scalar("loss", loss)
        else:
            tf.scalar_summary("loss", loss)

        # summary
        if parse_version(tf_version) >= parse_version('0.12.0'):
            summary_op = tf.summary.merge_all()
        else:
            summary_op = tf.merge_all_summaries()

    return loss, y, l1_loss





def build_fc_freq_4_30_NoTiedWeight_Small(
        x, 
        x_dim, 
        keep_prob, 
        is_training,
        gamma=1e-7, 
        activation='relu'):
    # FC1
    with tf.name_scope("FC1"):
        fc1_dim = 500
        W_fc1 = weight_variable([x_dim, fc1_dim])
        b_fc1 = weight_variable([fc1_dim])
        h_fc1 = tf.nn.elu(tf.matmul(x, W_fc1) + b_fc1)
        h_fc1_bn = batch_norm_contrib(h_fc1, is_training)

    # dropout
    #with tf.name_scope("Dropout1"):
    #    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # FC2
    with tf.name_scope("FC2"):
        fc2_dim = 200
        W_fc2 = weight_variable([fc1_dim, fc2_dim])
        b_fc2 = weight_variable([fc2_dim])
        h_fc2 = tf.nn.elu(tf.matmul(h_fc1_bn, W_fc2) + b_fc2)
        h_fc2_bn = batch_norm_contrib(h_fc2, is_training)

    # dropout
    #with tf.name_scope("Dropout2"):
    #    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    # FC3
    with tf.name_scope("FC3"):
        fc3_dim = 64
        W_fc3 = weight_variable([fc2_dim, fc3_dim])
        b_fc3 = weight_variable([fc3_dim])
        h_fc3 = tf.nn.elu(tf.matmul(h_fc2_bn, W_fc3) + b_fc3)

    # FC4
    with tf.name_scope("FCFeat"):
        fc4_dim = 64
        W_fc4 = weight_variable([fc3_dim, fc4_dim])
        b_fc4 = weight_variable([fc4_dim])
        h_fc4 = tf.nn.elu(tf.matmul(h_fc3, W_fc4) + b_fc4, name="feature")
        h_fc4_bn = batch_norm_contrib(h_fc4_bn, is_training)

    # FC5
    with tf.name_scope("FC5"):
        fc5_dim = 128
        W_fc5 = weight_variable([fc4_dim, fc5_dim])
        b_fc5 = weight_variable([fc5_dim])
        h_fc5 = tf.nn.elu(tf.matmul(h_fc4_bn, W_fc5) + b_fc5)
        h_fc5_bn = batch_norm_contrib(h_fc5_bn, is_training)


    with tf.name_scope("FC6"):
        fc6_dim = 200
        W_fc6 = weight_variable([fc3_dim, fc6_dim])
        b_fc6 = weight_variable([fc6_dim])
        h_fc6 = tf.nn.elu(tf.matmul(h_fc5_bn, W_fc6) + b_fc6)
        h_fc6_bn = batch_norm_contrib(h_fc6_bn, is_training)


    with tf.name_scope("FC7"):
        fc7_dim = 500
        W_fc7 = weight_variable([fc6_dim, fc7_dim])
        b_fc7 = weight_variable([fc7_dim])
        h_fc7 = tf.nn.elu(tf.matmul(h_fc6_bn, W_fc7) + b_fc7)
        h_fc7_bn = batch_norm_contrib(h_fc7_bn, is_training)

    # dropout
    #with tf.name_scope("Dropout7"):
    #    h_fc7_drop = tf.nn.dropout(h_fc7, keep_prob)

    # FC8
    with tf.name_scope("FC8"):
        fc8_dim = x_dim
        W_fc8 = weight_variable([fc7_dim, fc8_dim])
        b_fc8 = weight_variable([fc8_dim])
        h_fc8 = tf.matmul(h_fc7_bn, W_fc8) + b_fc8

    # LOSS 
    with tf.name_scope("loss"):
        y = h_fc8 + 1e-10
        l2_loss = tf.sqrt(tf.reduce_mean(tf.square(y-x)))
        #entropy_loss = - tf.reduce_mean(x * tf.log(y))

        #loss = entropy_loss + l2_loss
        loss = l2_loss 

        l1_loss_sum = tf.reduce_sum(tf.abs(W_fc1)) + \
                      tf.reduce_sum(tf.abs(W_fc2)) + \
                      tf.reduce_sum(tf.abs(W_fc3)) + \
                      tf.reduce_sum(tf.abs(W_fc6)) + \
                      tf.reduce_sum(tf.abs(W_fc7)) +  \
                      tf.reduce_sum(tf.abs(W_fc8))
        l1_loss = l1_loss_sum * gamma
        #loss += l1_loss

        tf_version = tf.__version__.rpartition('.')[0]
        if parse_version(tf_version) >= parse_version('0.12.0'):
            tf.summary.scalar("loss", loss)
        else:
            tf.scalar_summary("loss", loss)

        # summary
        if parse_version(tf_version) >= parse_version('0.12.0'):
            summary_op = tf.summary.merge_all()
        else:
            summary_op = tf.merge_all_summaries()

    return loss, y, l1_loss
