
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
from tf_utils import weight_variable, bias_variable, batch_norm_wrapper

def build_fc_freqSum_TiedWeight_NoBias(x, x_dim, keep_prob, gamma=1e-7):
    # FC1
    with tf.name_scope("FC1"):
        fc1_dim = 4096
        W_fc1 = weight_variable([x_dim, fc1_dim])
        h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1))


    # FC2
    with tf.name_scope("FC2"):
        fc2_dim = 2048
        W_fc2 = weight_variable([fc1_dim, fc2_dim])
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2))

    # dropout
    with tf.name_scope("Dropout1"):
        h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    # FC3
    with tf.name_scope("FC3"):
        fc3_dim = 1024
        W_fc3 = weight_variable([fc2_dim, fc3_dim])
        h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3))


    # FC4
    with tf.name_scope("FC4"):
        #fc4_dim = 256
        fc4_dim = 512
        W_fc4 = weight_variable([fc3_dim, fc4_dim])
        h_fc4 = tf.nn.relu(tf.matmul(h_fc3, W_fc4))

    # FC5
    with tf.name_scope("FC5"):
        fc5_dim = 256
        W_fc5 = weight_variable([fc4_dim, fc5_dim])
        h_fc5 = tf.nn.relu(tf.matmul(h_fc4, W_fc5))

    # FC6
    with tf.name_scope("FC6"):
        fc6_dim = 128 
        W_fc6 = weight_variable([fc5_dim, fc6_dim])
        h_fc6 = tf.nn.relu(tf.matmul(h_fc5, W_fc6))

    # dropout
    #with tf.name_scope("Dropout2"):
    #    h_fc6_drop = tf.nn.dropout(h_fc6, keep_prob)

    # FC7
    with tf.name_scope("FC7"):
        fc7_dim = 256
        W_fc7 = tf.transpose(W_fc6)
        h_fc7 = tf.nn.relu(tf.matmul(h_fc6, W_fc7))

    # FC8
    with tf.name_scope("FC8"):
        fc8_dim = 512
        W_fc8 = tf.transpose(W_fc5)
        h_fc8 = tf.nn.relu(tf.matmul(h_fc7, W_fc8))

    # FC9
    with tf.name_scope("FC9"):
        fc9_dim = 1024
        W_fc9 = tf.transpose(W_fc4)
        h_fc9 = tf.nn.relu(tf.matmul(h_fc8, W_fc9))


    # FC10
    with tf.name_scope("FC10"):
        fc10_dim = 2048
        W_fc10 = tf.transpose(W_fc3)
        h_fc10 = tf.nn.relu(tf.matmul(h_fc9, W_fc10))

    # dropout
    with tf.name_scope("Dropout3"):
        h_fc10_drop = tf.nn.dropout(h_fc10, keep_prob)

    # FC11
    with tf.name_scope("FC11"):
        fc11_dim = 4096
        W_fc11 = tf.transpose(W_fc2)
        h_fc11 = tf.nn.relu(tf.matmul(h_fc10_drop, W_fc11))

    # FC12
    with tf.name_scope("FC12"):
        fc12_dim = x_dim
        W_fc12 = tf.transpose(W_fc1)
        y = tf.matmul(h_fc11, W_fc12)


    # LOSS 
    with tf.name_scope("loss"):
        loss = tf.sqrt(tf.reduce_mean(tf.square(y-x)))
        #l1_loss = tf.reduce_sum(tf.abs(W_fc1)) + \
        #          tf.reduce_sum(tf.abs(W_fc2)) + \
        #          tf.reduce_sum(tf.abs(W_fc3)) + \
        #          tf.reduce_sum(tf.abs(W_fc4)) +  \
        #          tf.reduce_sum(tf.abs(W_fc5)) +  \
        #          tf.reduce_sum(tf.abs(W_fc6))
        #loss += gamma * l1_loss
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
    return loss, y


def build_fc_freqSum_TiedWeight_Big2(x, x_dim, keep_prob, gamma=1e-7):
    # FC1
    with tf.name_scope("FC1"):
        fc1_dim = 8192
        W_fc1 = weight_variable([x_dim, fc1_dim])
        b_fc1 = weight_variable([fc1_dim])
        h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)
        #h_fc1 = tf.matmul(x, W_fc1) + b_fc1


    # FC2
    with tf.name_scope("FC2"):
        fc2_dim = 4096
        W_fc2 = weight_variable([fc1_dim, fc2_dim])
        b_fc2 = weight_variable([fc2_dim])
        #h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    # dropout
    with tf.name_scope("Dropout1"):
        h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)


    # FC3
    with tf.name_scope("FC3"):
        fc3_dim = 2048
        W_fc3 = weight_variable([fc2_dim, fc3_dim])
        b_fc3 = weight_variable([fc3_dim])
        h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)
        #h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)

    # FC4
    with tf.name_scope("FC4"):
        fc4_dim = 1024
        W_fc4 = weight_variable([fc3_dim, fc4_dim])
        b_fc4 = weight_variable([fc4_dim])
        h_fc4 = tf.nn.relu(tf.matmul(h_fc3, W_fc4) + b_fc4)

    # FC5
    with tf.name_scope("FC5"):
        fc5_dim = 512
        W_fc5 = weight_variable([fc4_dim, fc5_dim])
        b_fc5 = weight_variable([fc5_dim])
        h_fc5 = tf.nn.relu(tf.matmul(h_fc4, W_fc5) + b_fc5)

    # FC6
    with tf.name_scope("FC6"):
        fc6_dim = 128 
        W_fc6 = weight_variable([fc5_dim, fc6_dim])
        b_fc6 = weight_variable([fc6_dim])
        h_fc6 = tf.nn.relu(tf.matmul(h_fc5, W_fc6) + b_fc6)

    # dropout
    #with tf.name_scope("Dropout2"):
    #    h_fc6_drop = tf.nn.dropout(h_fc6, keep_prob)

    # FC7
    with tf.name_scope("FC7"):
        fc7_dim = 512
        W_fc7 = tf.transpose(W_fc6)
        b_fc7 = weight_variable([fc7_dim])
        h_fc7 = tf.nn.relu(tf.matmul(h_fc6, W_fc7) + b_fc7)

    # FC8
    with tf.name_scope("FC8"):
        fc8_dim = 1024
        W_fc8 = tf.transpose(W_fc5)
        b_fc8 = weight_variable([fc8_dim])
        h_fc8 = tf.nn.relu(tf.matmul(h_fc7, W_fc8) + b_fc8)

    # FC9
    with tf.name_scope("FC9"):
        fc9_dim = 2048
        W_fc9 = tf.transpose(W_fc4)
        b_fc9 = weight_variable([fc9_dim])
        h_fc9 = tf.nn.relu(tf.matmul(h_fc8, W_fc9) + b_fc9)


    # FC10
    with tf.name_scope("FC10"):
        fc10_dim = 4096
        W_fc10 = tf.transpose(W_fc3)
        b_fc10 = weight_variable([fc10_dim])
        h_fc10 = tf.nn.relu(tf.matmul(h_fc9, W_fc10) + b_fc10)

    # dropout
    with tf.name_scope("Dropout3"):
        h_fc10_drop = tf.nn.dropout(h_fc10, keep_prob)

    # FC11
    with tf.name_scope("FC11"):
        fc11_dim = 8192
        W_fc11 = tf.transpose(W_fc2)
        b_fc11 = weight_variable([fc11_dim])
        h_fc11 = tf.nn.relu(tf.matmul(h_fc10_drop, W_fc11) + b_fc11)

    # FC12
    with tf.name_scope("FC12"):
        fc12_dim = x_dim
        W_fc12 = tf.transpose(W_fc1)
        b_fc12 = weight_variable([fc12_dim])
        y = tf.sigmoid(tf.matmul(h_fc11, W_fc12) + b_fc12)


    # LOSS 
    with tf.name_scope("loss"):
        #loss = tf.reduce_mean(tf.squared_difference(y, x))
        l2_loss = tf.sqrt(tf.reduce_mean(tf.square(y-x)))
        entropy_loss = - tf.reduce_mean(x * tf.log(y))

        loss = entropy_loss

        #l1_loss = tf.reduce_sum(tf.abs(W_fc1)) + \
        #          tf.reduce_sum(tf.abs(W_fc2)) + \
        #          tf.reduce_sum(tf.abs(W_fc3)) + \
        #          tf.reduce_sum(tf.abs(W_fc4)) +  \
        #          tf.reduce_sum(tf.abs(W_fc5)) +  \
        #          tf.reduce_sum(tf.abs(W_fc6))
        ##loss += gamma * l1_loss
        if parse_version(tf_version) >= parse_version('0.12.0'):
            tf.summary.scalar("loss", loss)
        else:
            tf.scalar_summary("loss", loss)

        # summary
        if parse_version(tf_version) >= parse_version('0.12.0'):
            summary_op = tf.summary.merge_all()
        else:
            summary_op = tf.merge_all_summaries()

    return loss, y, tf.constant(0) 



def build_fc_freq_5_TiedWeight_Small(x, x_dim, keep_prob, is_training,
                                        gamma=1e-7, activation='relu'):
    # FC1
    with tf.name_scope("FC1"):
        fc1_dim = 1024
        W_fc1 = weight_variable([x_dim, fc1_dim])
        b_fc1 = weight_variable([fc1_dim])
        h_fc1 = tf.nn.elu(tf.matmul(x, W_fc1) + b_fc1)
        h_fc1 = batch_norm_wrapper(h_fc1, is_training)

    # dropout
    #with tf.name_scope("Dropout1"):
    #    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # FC2
    with tf.name_scope("FC2"):
        #fc2_dim = 4096
        fc2_dim = 512
        W_fc2 = weight_variable([fc1_dim, fc2_dim])
        b_fc2 = weight_variable([fc2_dim])
        h_fc2 = tf.nn.elu(tf.matmul(h_fc1, W_fc2) + b_fc2)
        h_fc2 = batch_norm_wrapper(h_fc2, is_training)

    # dropout
    #with tf.name_scope("Dropout2"):
    #    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    # FC3
    with tf.name_scope("FC3"):
        fc3_dim = 256
        W_fc3 = weight_variable([fc2_dim, fc3_dim])
        b_fc3 = weight_variable([fc3_dim])
        #h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)
        h_fc3 = tf.nn.elu(tf.matmul(h_fc2, W_fc3) + b_fc3)
        h_fc3 = batch_norm_wrapper(h_fc3, is_training)

    # FC4
    with tf.name_scope("FC4"):
        fc4_dim = 64
        W_fc4 = weight_variable([fc3_dim, fc4_dim])
        b_fc4 = weight_variable([fc4_dim])
        h_fc4 = tf.nn.elu(tf.matmul(h_fc3, W_fc4) + b_fc4)
        h_fc4 = batch_norm_wrapper(h_fc4, is_training)

    # FC5
    with tf.name_scope("FC5"):
        fc5_dim = 256
        #W_fc5 = weight_variable([fc4_dim, fc5_dim])
        W_fc5 = tf.transpose(W_fc4)
        b_fc5 = weight_variable([fc5_dim])
        h_fc5 = tf.nn.elu(tf.matmul(h_fc4, W_fc5) + b_fc5)
        h_fc5 = batch_norm_wrapper(h_fc5, is_training)

    with tf.name_scope("FC6"):
        fc6_dim = 512
        W_fc6 = tf.transpose(W_fc3)
        b_fc6 = weight_variable([fc6_dim])
        h_fc6 = tf.nn.elu(tf.matmul(h_fc5, W_fc6) + b_fc6)
        h_fc6 = batch_norm_wrapper(h_fc6, is_training)


    with tf.name_scope("FC7"):
        fc7_dim = 1024
        W_fc7 = tf.transpose(W_fc2)
        b_fc7 = weight_variable([fc7_dim])
        h_fc7 = tf.nn.elu(tf.matmul(h_fc6, W_fc7) + b_fc7)
        h_fc7 = batch_norm_wrapper(h_fc7, is_training)

    # dropout
    #with tf.name_scope("Dropout7"):
    #    h_fc7_drop = tf.nn.dropout(h_fc7, keep_prob)

    # FC8
    with tf.name_scope("FC8"):
        fc8_dim = x_dim
        #W_fc8 = weight_variable([fc7_dim, fc8_dim])
        W_fc8 = tf.transpose(W_fc1)
        b_fc8 = weight_variable([fc8_dim])
        h_fc8 = tf.matmul(h_fc7, W_fc8) + b_fc8

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





def build_fc_freq_4_30_TiedWeight_Small(x, x_dim, keep_prob, is_training,
                                        gamma=1e-7, activation='relu'):
    # FC1
    with tf.name_scope("FC1"):
        fc1_dim = 512
        W_fc1 = weight_variable([x_dim, fc1_dim])
        b_fc1 = weight_variable([fc1_dim])
        h_fc1 = tf.nn.elu(tf.matmul(x, W_fc1) + b_fc1)
        h_fc1 = batch_norm_wrapper(h_fc1, is_training)

    # dropout
    #with tf.name_scope("Dropout1"):
    #    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # FC2
    with tf.name_scope("FC2"):
        #fc2_dim = 4096
        fc2_dim = 256
        W_fc2 = weight_variable([fc1_dim, fc2_dim])
        b_fc2 = weight_variable([fc2_dim])
        h_fc2 = tf.nn.elu(tf.matmul(h_fc1, W_fc2) + b_fc2)
        h_fc2 = batch_norm_wrapper(h_fc2, is_training)

    # dropout
    #with tf.name_scope("Dropout2"):
    #    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    # FC3
    with tf.name_scope("FC3"):
        fc3_dim = 128
        W_fc3 = weight_variable([fc2_dim, fc3_dim])
        b_fc3 = weight_variable([fc3_dim])
        #h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)
        h_fc3 = tf.nn.elu(tf.matmul(h_fc2, W_fc3) + b_fc3)
        h_fc3 = batch_norm_wrapper(h_fc3, is_training)

    # FC4
    with tf.name_scope("FC4"):
        fc4_dim = 64
        W_fc4 = weight_variable([fc3_dim, fc4_dim])
        b_fc4 = weight_variable([fc4_dim])
        h_fc4 = tf.nn.elu(tf.matmul(h_fc3, W_fc4) + b_fc4)
        h_fc4 = batch_norm_wrapper(h_fc4, is_training)

    # FC5
    with tf.name_scope("FC5"):
        fc5_dim = 128
        #W_fc5 = weight_variable([fc4_dim, fc5_dim])
        W_fc5 = tf.transpose(W_fc4)
        b_fc5 = weight_variable([fc5_dim])
        h_fc5 = tf.nn.elu(tf.matmul(h_fc4, W_fc5) + b_fc5)
        h_fc5 = batch_norm_wrapper(h_fc5, is_training)

    with tf.name_scope("FC6"):
        fc6_dim = 256
        W_fc6 = tf.transpose(W_fc3)
        b_fc6 = weight_variable([fc6_dim])
        h_fc6 = tf.nn.elu(tf.matmul(h_fc5, W_fc6) + b_fc6)
        h_fc6 = batch_norm_wrapper(h_fc6, is_training)


    with tf.name_scope("FC7"):
        fc7_dim = 512
        W_fc7 = tf.transpose(W_fc2)
        b_fc7 = weight_variable([fc7_dim])
        h_fc7 = tf.nn.elu(tf.matmul(h_fc6, W_fc7) + b_fc7)
        h_fc7 = batch_norm_wrapper(h_fc7, is_training)

    # dropout
    #with tf.name_scope("Dropout7"):
    #    h_fc7_drop = tf.nn.dropout(h_fc7, keep_prob)

    # FC8
    with tf.name_scope("FC8"):
        fc8_dim = x_dim
        #W_fc8 = weight_variable([fc7_dim, fc8_dim])
        W_fc8 = tf.transpose(W_fc1)
        b_fc8 = weight_variable([fc8_dim])
        h_fc8 = tf.matmul(h_fc7, W_fc8) + b_fc8

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





def build_fc_freq_4_30_NoTiedWeight_Small(x, x_dim, keep_prob, gamma=1e-7, activation='relu'):
    # FC1
    with tf.name_scope("FC1"):
        fc1_dim = 500
        W_fc1 = weight_variable([x_dim, fc1_dim])
        b_fc1 = weight_variable([fc1_dim])
        #h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)
        h_fc1 = tf.nn.elu(tf.matmul(x, W_fc1) + b_fc1)

    # dropout
    #with tf.name_scope("Dropout1"):
    #    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # FC2
    with tf.name_scope("FC2"):
        #fc2_dim = 4096
        fc2_dim = 200
        W_fc2 = weight_variable([fc1_dim, fc2_dim])
        b_fc2 = weight_variable([fc2_dim])
        #h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        #h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
        h_fc2 = tf.nn.elu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    # dropout
    #with tf.name_scope("Dropout2"):
    #    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    # FC3
    with tf.name_scope("FC3"):
        fc3_dim = 64
        W_fc3 = weight_variable([fc2_dim, fc3_dim])
        b_fc3 = weight_variable([fc3_dim])
        h_fc3 = tf.nn.elu(tf.matmul(h_fc2, W_fc3) + b_fc3)

    # FC4
    with tf.name_scope("FC4"):
        fc4_dim = 64
        W_fc4 = weight_variable([fc3_dim, fc4_dim])
        b_fc4 = weight_variable([fc4_dim])
        h_fc4 = tf.nn.elu(tf.matmul(h_fc3, W_fc4) + b_fc4)

    # FC5
    with tf.name_scope("FC5"):
        fc5_dim = 128
        W_fc5 = weight_variable([fc4_dim, fc5_dim])
        b_fc5 = weight_variable([fc5_dim])
        h_fc5 = tf.nn.elu(tf.matmul(h_fc4, W_fc5) + b_fc5)


    with tf.name_scope("FC6"):
        fc6_dim = 200
        W_fc6 = weight_variable([fc3_dim, fc6_dim])
        b_fc6 = weight_variable([fc6_dim])
        h_fc6 = tf.nn.elu(tf.matmul(h_fc5, W_fc6) + b_fc6)


    with tf.name_scope("FC7"):
        fc7_dim = 500
        W_fc7 = weight_variable([fc6_dim, fc7_dim])
        b_fc7 = weight_variable([fc7_dim])
        h_fc7 = tf.nn.elu(tf.matmul(h_fc6, W_fc7) + b_fc7)

    # dropout
    #with tf.name_scope("Dropout7"):
    #    h_fc7_drop = tf.nn.dropout(h_fc7, keep_prob)

    # FC8
    with tf.name_scope("FC8"):
        fc8_dim = x_dim
        W_fc8 = weight_variable([fc7_dim, fc8_dim])
        b_fc8 = weight_variable([fc8_dim])
        h_fc8 = tf.matmul(h_fc7, W_fc8) + b_fc8

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






def build_fc_freqSum_NoTiedWeight_Medium(x, x_dim, keep_prob, gamma=1e-7,
        activation='relu'):
    # FC1
    with tf.name_scope("FC1"):
        fc1_dim = 4096
        W_fc1 = weight_variable([x_dim, fc1_dim])
        b_fc1 = weight_variable([fc1_dim])
        h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

    # dropout
    #with tf.name_scope("Dropout1"):
    #    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # FC2
    with tf.name_scope("FC2"):
        #fc2_dim = 4096
        fc2_dim = 1024
        W_fc2 = weight_variable([fc1_dim, fc2_dim])
        b_fc2 = weight_variable([fc2_dim])
        #h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    # dropout
    #with tf.name_scope("Dropout2"):
    #    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    # FC3
    with tf.name_scope("FC3"):
        fc3_dim = 256
        W_fc3 = weight_variable([fc2_dim, fc3_dim])
        b_fc3 = weight_variable([fc3_dim])
        #h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)
        h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)

    # FC4
    with tf.name_scope("FC4"):
        fc4_dim = 64
        W_fc4 = weight_variable([fc3_dim, fc4_dim])
        b_fc4 = weight_variable([fc4_dim])
        if activation == 'softmax':
            h_fc4 = tf.nn.softmax(tf.matmul(h_fc3, W_fc4) + b_fc4)
        elif activation == 'linear':
            h_fc4 = tf.matmul(h_fc3, W_fc4) + b_fc4
        else:
            h_fc4 = tf.nn.relu(tf.matmul(h_fc3, W_fc4) + b_fc4)

    # FC5
    with tf.name_scope("FC5"):
        fc5_dim = 256 
        W_fc5 = weight_variable([fc4_dim, fc5_dim])
        b_fc5 = weight_variable([fc5_dim])
        h_fc5 = tf.nn.relu(tf.matmul(h_fc4, W_fc5) + b_fc5)



    with tf.name_scope("FC6"):
        fc6_dim = 1024
        W_fc6 = weight_variable([fc3_dim, fc6_dim])
        #W_fc6 = tf.transpose(W_fc3)
        b_fc6 = weight_variable([fc6_dim])
        #h_fc6 = tf.nn.relu(tf.matmul(h_fc5_drop, W_fc6) + b_fc6)
        h_fc6 = tf.nn.relu(tf.matmul(h_fc5, W_fc6) + b_fc6)


    with tf.name_scope("FC7"):
        fc7_dim = 4096
        #W_fc7 = tf.transpose(W_fc2)
        W_fc7 = weight_variable([fc6_dim, fc7_dim])
        b_fc7 = weight_variable([fc7_dim])
        h_fc7 = tf.nn.relu(tf.matmul(h_fc6, W_fc7) + b_fc7)
        #h_fc7 = tf.nn.relu(tf.matmul(h_fc6_drop, W_fc7) + b_fc7)

    # dropout
    #with tf.name_scope("Dropout7"):
    #    h_fc7_drop = tf.nn.dropout(h_fc7, keep_prob)

    # FC8
    with tf.name_scope("FC8"):
        fc8_dim = x_dim
        #W_fc8 = tf.transpose(W_fc1)
        W_fc8 = weight_variable([fc7_dim, fc8_dim])
        b_fc8 = weight_variable([fc8_dim])
        #h_fc8 = tf.nn.sigmoid(tf.matmul(h_fc7_drop, W_fc8) + b_fc8)
        #h_fc8 = tf.nn.relu(tf.matmul(h_fc7, W_fc8) + b_fc8)
        h_fc8 = tf.matmul(h_fc7, W_fc8) + b_fc8

    # FC9
    #with tf.name_scope("FC9"):
    #    fc9_dim =  512
    #    W_fc9 = tf.transpose(W_fc4)
    #    b_fc9 = weight_variable([fc9_dim])
    #    h_fc9 = tf.nn.relu(tf.matmul(h_fc8, W_fc9) + b_fc9)


    # FC10
    #with tf.name_scope("FC10"):
    #    fc10_dim = 1024
    #    W_fc10 = tf.transpose(W_fc3)
    #    b_fc10 = weight_variable([fc10_dim])
    #    h_fc10 = tf.nn.relu(tf.matmul(h_fc9, W_fc10) + b_fc10)

    # dropout
    #with tf.name_scope("Dropout3"):
    #    h_fc10_drop = tf.nn.dropout(h_fc10, keep_prob)

    # FC11
    #with tf.name_scope("FC11"):
    #    fc11_dim = 2048
    #    W_fc11 = tf.transpose(W_fc2)
    #    b_fc11 = weight_variable([fc11_dim])
    #    h_fc11 = tf.nn.relu(tf.matmul(h_fc10_drop, W_fc11) + b_fc11)
#
#    # FC12
#    with tf.name_scope("FC12"):
#        fc12_dim = x_dim
#        W_fc12 = tf.transpose(W_fc1)
#        b_fc12 = weight_variable([fc12_dim])
#        y = tf.sigmoid(tf.matmul(h_fc11, W_fc12) + b_fc12)
#

    # LOSS 
    with tf.name_scope("loss"):
        y = h_fc8 + 1e-10
        #loss = tf.reduce_mean(tf.squared_difference(y, x))
        l2_loss = tf.sqrt(tf.reduce_mean(tf.square(y-x)))
        #entropy_loss = - tf.reduce_mean(x * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
        entropy_loss = - tf.reduce_mean(x * tf.log(y))

        #loss = entropy_loss + l2_loss
        loss = l2_loss 

        l1_loss_sum = tf.reduce_sum(tf.abs(W_fc1)) + \
                      tf.reduce_sum(tf.abs(W_fc2)) + \
                      tf.reduce_sum(tf.abs(W_fc3))  + \
                      tf.reduce_sum(tf.abs(W_fc4)) + \
                      tf.reduce_sum(tf.abs(W_fc5)) +  \
                      tf.reduce_sum(tf.abs(W_fc6)) + \
                      tf.reduce_sum(tf.abs(W_fc7)) +  \
                      tf.reduce_sum(tf.abs(W_fc8))
        l1_loss = l1_loss_sum * gamma
        loss += l1_loss

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

    return loss, y,  l1_loss





def build_fc_freqSum_NoTiedWeight_Small(x, x_dim, keep_prob, gamma=1e-7):
    # FC1
    with tf.name_scope("FC1"):
        fc1_dim = 4096
        W_fc1 = weight_variable([x_dim, fc1_dim])
        b_fc1 = weight_variable([fc1_dim])
        h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

    # dropout
    #with tf.name_scope("Dropout1"):
    #    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # FC2
    with tf.name_scope("FC2"):
        #fc2_dim = 4096
        fc2_dim = 1024
        W_fc2 = weight_variable([fc1_dim, fc2_dim])
        b_fc2 = weight_variable([fc2_dim])
        #h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    # dropout
    #with tf.name_scope("Dropout2"):
    #    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    # FC3
    with tf.name_scope("FC3"):
        fc3_dim = 64
        W_fc3 = weight_variable([fc2_dim, fc3_dim])
        b_fc3 = weight_variable([fc3_dim])
        #h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)
        h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)


    with tf.name_scope("FC6"):
        fc6_dim = 1024
        W_fc6 = weight_variable([fc3_dim, fc6_dim])
        #W_fc6 = tf.transpose(W_fc3)
        b_fc6 = weight_variable([fc6_dim])
        #h_fc6 = tf.nn.relu(tf.matmul(h_fc5_drop, W_fc6) + b_fc6)
        h_fc6 = tf.nn.relu(tf.matmul(h_fc3, W_fc6) + b_fc6)


    with tf.name_scope("FC7"):
        fc7_dim = 4096
        #W_fc7 = tf.transpose(W_fc2)
        W_fc7 = weight_variable([fc6_dim, fc7_dim])
        b_fc7 = weight_variable([fc7_dim])
        h_fc7 = tf.nn.relu(tf.matmul(h_fc6, W_fc7) + b_fc7)
        #h_fc7 = tf.nn.relu(tf.matmul(h_fc6_drop, W_fc7) + b_fc7)

    # dropout
    #with tf.name_scope("Dropout7"):
    #    h_fc7_drop = tf.nn.dropout(h_fc7, keep_prob)

    # FC8
    with tf.name_scope("FC8"):
        fc8_dim = x_dim
        #W_fc8 = tf.transpose(W_fc1)
        W_fc8 = weight_variable([fc7_dim, fc8_dim])
        b_fc8 = weight_variable([fc8_dim])
        #h_fc8 = tf.nn.sigmoid(tf.matmul(h_fc7_drop, W_fc8) + b_fc8)
        h_fc8 = tf.nn.relu(tf.matmul(h_fc7, W_fc8) + b_fc8)

    # FC9
    #with tf.name_scope("FC9"):
    #    fc9_dim =  512
    #    W_fc9 = tf.transpose(W_fc4)
    #    b_fc9 = weight_variable([fc9_dim])
    #    h_fc9 = tf.nn.relu(tf.matmul(h_fc8, W_fc9) + b_fc9)


    # FC10
    #with tf.name_scope("FC10"):
    #    fc10_dim = 1024
    #    W_fc10 = tf.transpose(W_fc3)
    #    b_fc10 = weight_variable([fc10_dim])
    #    h_fc10 = tf.nn.relu(tf.matmul(h_fc9, W_fc10) + b_fc10)

    # dropout
    #with tf.name_scope("Dropout3"):
    #    h_fc10_drop = tf.nn.dropout(h_fc10, keep_prob)

    # FC11
    #with tf.name_scope("FC11"):
    #    fc11_dim = 2048
    #    W_fc11 = tf.transpose(W_fc2)
    #    b_fc11 = weight_variable([fc11_dim])
    #    h_fc11 = tf.nn.relu(tf.matmul(h_fc10_drop, W_fc11) + b_fc11)
#
#    # FC12
#    with tf.name_scope("FC12"):
#        fc12_dim = x_dim
#        W_fc12 = tf.transpose(W_fc1)
#        b_fc12 = weight_variable([fc12_dim])
#        y = tf.sigmoid(tf.matmul(h_fc11, W_fc12) + b_fc12)
#

    # LOSS 
    with tf.name_scope("loss"):
        y = h_fc8
        #loss = tf.reduce_mean(tf.squared_difference(y, x))
        l2_loss = tf.sqrt(tf.reduce_mean(tf.square(y-x)))
        entropy_loss = - tf.reduce_mean(x * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

        loss = entropy_loss + l2_loss
        #loss = l2_loss 

        l1_loss_sum = tf.reduce_sum(tf.abs(W_fc1)) + \
                      tf.reduce_sum(tf.abs(W_fc2)) + \
                      tf.reduce_sum(tf.abs(W_fc3)) 
                  #tf.reduce_sum(tf.abs(W_fc4))
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

    return loss, y, entropy_loss




def build_fc_freqSum_NoTiedWeight_Big(x, x_dim, keep_prob, gamma=1e-7,
        activation='relu'):
    # FC1
    with tf.name_scope("FC1"):
        #fc1_dim = 8192
        fc1_dim = 4096
        W_fc1 = weight_variable([x_dim, fc1_dim])
        b_fc1 = weight_variable([fc1_dim])
        h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

    # dropout
    #with tf.name_scope("Dropout1"):
    #    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # FC2
    with tf.name_scope("FC2"):
        fc2_dim = 2048
        W_fc2 = weight_variable([fc1_dim, fc2_dim])
        b_fc2 = weight_variable([fc2_dim])
        #h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    # dropout
    #with tf.name_scope("Dropout2"):
    #    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    # FC3
    with tf.name_scope("FC3"):
        fc3_dim = 1024
        W_fc3 = weight_variable([fc2_dim, fc3_dim])
        b_fc3 = weight_variable([fc3_dim])
        #h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)
        h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)

    # dropout
    #with tf.name_scope("Dropout3"):
    #    h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)

    # FC4
    with tf.name_scope("FC4"):
        fc4_dim = 512
        W_fc4 = weight_variable([fc3_dim, fc4_dim])
        b_fc4 = weight_variable([fc4_dim])
        #h_fc4 = tf.nn.relu(tf.matmul(h_fc3_drop, W_fc4) + b_fc4)
        h_fc4 = tf.nn.relu(tf.matmul(h_fc3, W_fc4) + b_fc4)

    # dropout
    #with tf.name_scope("Dropout4"):
    #    h_fc4_drop = tf.nn.dropout(h_fc4, keep_prob)

    # FC5
    with tf.name_scope("FC5"):
        fc5_dim = 256
        W_fc5 = weight_variable([fc4_dim, fc5_dim])
        b_fc5 = weight_variable([fc5_dim])
        h_fc5 = tf.nn.relu(tf.matmul(h_fc4, W_fc5) + b_fc5)

    # dropout
    #with tf.name_scope("Dropout5"):
    #    h_fc5_drop = tf.nn.dropout(h_fc5, keep_prob)

    # FC6
    with tf.name_scope("FC6"):
        fc6_dim = 64
        W_fc6 = weight_variable([fc5_dim, fc6_dim])
        #W_fc6 = tf.transpose(W_fc3)
        b_fc6 = weight_variable([fc6_dim])
        if activation == 'softmax':
            h_fc6 = tf.nn.softmax(tf.matmul(h_fc5, W_fc6) + b_fc6)
        elif activation == 'linear':
            h_fc6 = tf.matmul(h_fc5, W_fc6) + b_fc6
        else:
            h_fc6 = tf.nn.relu(tf.matmul(h_fc5, W_fc6) + b_fc6)

    # dropout
    #with tf.name_scope("Dropout6"):
    #    h_fc6_drop = tf.nn.dropout(h_fc6, keep_prob)

    # FC7
    with tf.name_scope("FC7"):
        fc7_dim = 256
        #W_fc7 = tf.transpose(W_fc2)
        W_fc7 = weight_variable([fc6_dim, fc7_dim])
        b_fc7 = weight_variable([fc7_dim])
        h_fc7 = tf.nn.relu(tf.matmul(h_fc6, W_fc7) + b_fc7)
        #h_fc7 = tf.nn.relu(tf.matmul(h_fc6_drop, W_fc7) + b_fc7)

    # dropout
    #with tf.name_scope("Dropout7"):
    #    h_fc7_drop = tf.nn.dropout(h_fc7, keep_prob)

    # FC8
    with tf.name_scope("FC8"):
        fc8_dim = 512
        #W_fc8 = tf.transpose(W_fc1)
        W_fc8 = weight_variable([fc7_dim, fc8_dim])
        b_fc8 = weight_variable([fc8_dim])
        #h_fc8 = tf.nn.relu(tf.matmul(h_fc7_drop, W_fc8) + b_fc8)
        h_fc8 = tf.nn.relu(tf.matmul(h_fc7, W_fc8) + b_fc8)

    # dropout
    #with tf.name_scope("Dropout8"):
    #    h_fc8_drop = tf.nn.dropout(h_fc8, keep_prob)

    # FC9
    with tf.name_scope("FC9"):
        fc9_dim = 1024
        #W_fc9 = tf.transpose(W_fc4)
        W_fc9 = weight_variable([fc8_dim, fc9_dim])
        b_fc9 = weight_variable([fc9_dim])
        h_fc9 = tf.nn.relu(tf.matmul(h_fc8, W_fc9) + b_fc9)
        #h_fc9 = tf.nn.relu(tf.matmul(h_fc8_drop, W_fc9) + b_fc9)

    # dropout
    #with tf.name_scope("Dropout9"):
    #    h_fc9_drop = tf.nn.dropout(h_fc9, keep_prob)

    # FC10
    with tf.name_scope("FC10"):
        fc10_dim = 2048
        #W_fc10 = tf.transpose(W_fc3)
        W_fc10 = weight_variable([fc9_dim, fc10_dim])
        b_fc10 = weight_variable([fc10_dim])
        #h_fc10 = tf.nn.relu(tf.matmul(h_fc9_drop, W_fc10) + b_fc10)
        h_fc10 = tf.nn.relu(tf.matmul(h_fc9, W_fc10) + b_fc10)

    # dropout
    #with tf.name_scope("Dropout10"):
    #    h_fc10_drop = tf.nn.dropout(h_fc10, keep_prob)

    # FC11
    with tf.name_scope("FC11"):
        fc11_dim = 4096
        #W_fc11 = tf.transpose(W_fc2)
        W_fc11 = weight_variable([fc10_dim, fc11_dim])
        b_fc11 = weight_variable([fc11_dim])
        #h_fc11 = tf.nn.relu(tf.matmul(h_fc10_drop, W_fc11) + b_fc11)
        h_fc11 = tf.nn.relu(tf.matmul(h_fc10, W_fc11) + b_fc11)
#
#    # FC12
    with tf.name_scope("FC12"):
        fc12_dim = x_dim
        #W_fc12 = tf.transpose(W_fc1)
        W_fc12 = weight_variable([fc11_dim, fc12_dim])
        b_fc12 = weight_variable([fc12_dim])
        y = tf.matmul(h_fc11, W_fc12) + b_fc12
#

    # LOSS 
    with tf.name_scope("loss"):
        l2_loss = tf.sqrt(tf.reduce_mean(tf.square(y-x)))
        #entropy_loss = - tf.reduce_mean(x * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

        #loss = entropy_loss + l2_loss
        loss = l2_loss 

        l1_loss = tf.reduce_sum(tf.abs(W_fc1)) + \
                  tf.reduce_sum(tf.abs(W_fc2)) + \
                  tf.reduce_sum(tf.abs(W_fc3)) + \
                  tf.reduce_sum(tf.abs(W_fc4)) + \
                  tf.reduce_sum(tf.abs(W_fc5)) +  \
                  tf.reduce_sum(tf.abs(W_fc6))
        loss += gamma * l1_loss
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
    return loss, y, gamma * l1_loss


def build_fc_freqSum_NoTiedWeight_Tiny(x, x_dim, keep_prob, gamma=1e-7):
    # FC1
    with tf.name_scope("FC1"):
        fc1_dim = 64
        W_fc1 = weight_variable([x_dim, fc1_dim])
        b_fc1 = weight_variable([fc1_dim])
        h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

    #with tf.name_scope("Dropout1"):
    #    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


    # FC2
    with tf.name_scope("FC2"):
        fc2_dim = x_dim
        W_fc2 = weight_variable([fc1_dim, fc2_dim])
        b_fc2 = weight_variable([fc2_dim])
        y = tf.matmul(h_fc1, W_fc2) + b_fc2

    # LOSS 
    with tf.name_scope("loss"):
        l2_loss = tf.sqrt(tf.reduce_mean(tf.square(y-x)))
        loss = l2_loss

        l1_loss = tf.reduce_sum(tf.abs(W_fc1)) + \
                  tf.reduce_sum(tf.abs(W_fc2)) 
        loss += gamma * l1_loss

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
    return loss, y, gamma * l1_loss



def build_fc_freqSum_TiedWeight_Big(x, x_dim, keep_prob, gamma=1e-7,
        activation='relu'):
    # FC1
    with tf.name_scope("FC1"):
        #fc1_dim = 8192
        fc1_dim = 4096
        W_fc1 = weight_variable([x_dim, fc1_dim])
        b_fc1 = weight_variable([fc1_dim])
        h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

    # dropout
    #with tf.name_scope("Dropout1"):
    #    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # FC2
    with tf.name_scope("FC2"):
        #fc2_dim = 4096
        fc2_dim = 2048
        W_fc2 = weight_variable([fc1_dim, fc2_dim])
        b_fc2 = weight_variable([fc2_dim])
        #h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    # dropout
    #with tf.name_scope("Dropout2"):
    #    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    # FC3
    with tf.name_scope("FC3"):
        fc3_dim = 1024
        W_fc3 = weight_variable([fc2_dim, fc3_dim])
        b_fc3 = weight_variable([fc3_dim])
        #h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)
        h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)

    # dropout
    #with tf.name_scope("Dropout3"):
    #    h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)

    # FC4
    with tf.name_scope("FC4"):
        fc4_dim = 512
        W_fc4 = weight_variable([fc3_dim, fc4_dim])
        b_fc4 = weight_variable([fc4_dim])
        h_fc4 = tf.nn.relu(tf.matmul(h_fc3, W_fc4) + b_fc4)

    # dropout
    #with tf.name_scope("Dropout4"):
    #    h_fc4_drop = tf.nn.dropout(h_fc4, keep_prob)

    # FC5
    with tf.name_scope("FC5"):
        fc5_dim = 256
        W_fc5 = weight_variable([fc4_dim, fc5_dim])
        b_fc5 = weight_variable([fc5_dim])
        h_fc5 = tf.nn.relu(tf.matmul(h_fc4, W_fc5) + b_fc5)

    # dropout
    #with tf.name_scope("Dropout5"):
    #    h_fc5_drop = tf.nn.dropout(h_fc5, keep_prob)

    # FC6
    with tf.name_scope("FC6"):
        fc6_dim = 64
        W_fc6 = weight_variable([fc5_dim, fc6_dim])
        #W_fc6 = tf.transpose(W_fc4)
        b_fc6 = weight_variable([fc6_dim])
        if activation == 'softmax':
            h_fc6 = tf.nn.softmax(tf.matmul(h_fc5, W_fc6) + b_fc6)
        else:
            h_fc6 = tf.nn.relu(tf.matmul(h_fc5, W_fc6) + b_fc6)

    # dropout
    #with tf.name_scope("Dropout6"):
    #    h_fc6_drop = tf.nn.dropout(h_fc6, keep_prob)

    # FC7
    with tf.name_scope("FC7"):
        fc7_dim = 256
        W_fc7 = tf.transpose(W_fc6)
        b_fc7 = weight_variable([fc7_dim])
        #h_fc7 = tf.nn.relu(tf.matmul(h_fc6, W_fc7) + b_fc7)
        h_fc7 = tf.nn.relu(tf.matmul(h_fc6, W_fc7) + b_fc7)

    # dropout
    #with tf.name_scope("Dropout7"):
    #    h_fc7_drop = tf.nn.dropout(h_fc7, keep_prob)

    # FC8
    with tf.name_scope("FC8"):
        fc8_dim = 512
        W_fc8 = tf.transpose(W_fc5)
        b_fc8 = weight_variable([fc8_dim])
        #h_fc8 = tf.nn.sigmoid(tf.matmul(h_fc7, W_fc8) + b_fc8)
        h_fc8 = tf.nn.relu(tf.matmul(h_fc7, W_fc8) + b_fc8)

    # FC9
    with tf.name_scope("FC9"):
        fc9_dim =  1024
        W_fc9 = tf.transpose(W_fc4)
        b_fc9 = weight_variable([fc9_dim])
        h_fc9 = tf.nn.relu(tf.matmul(h_fc8, W_fc9) + b_fc9)


    # FC10
    with tf.name_scope("FC10"):
        fc10_dim = 2048
        W_fc10 = tf.transpose(W_fc3)
        b_fc10 = weight_variable([fc10_dim])
        h_fc10 = tf.nn.relu(tf.matmul(h_fc9, W_fc10) + b_fc10)

    # dropout
    #with tf.name_scope("Dropout3"):
    #    h_fc10_drop = tf.nn.dropout(h_fc10, keep_prob)

    # FC11
    with tf.name_scope("FC11"):
        fc11_dim = 4096
        W_fc11 = tf.transpose(W_fc2)
        b_fc11 = weight_variable([fc11_dim])
        h_fc11 = tf.nn.relu(tf.matmul(h_fc10, W_fc11) + b_fc11)

#    # FC12
    with tf.name_scope("FC12"):
        fc12_dim = x_dim
        W_fc12 = tf.transpose(W_fc1)
        b_fc12 = weight_variable([fc12_dim])
        y = tf.matmul(h_fc11, W_fc12) + b_fc12
#

    # LOSS 
    with tf.name_scope("loss"):
        #loss = tf.reduce_mean(tf.squared_difference(y, x))
        l2_loss = tf.sqrt(tf.reduce_mean(tf.square(y-x)))
        #entropy_loss = - tf.reduce_mean(x * tf.log(y + 1e-10))
        #loss = entropy_loss + l2_loss

        loss = l2_loss

        l1_loss = tf.reduce_sum(tf.abs(W_fc1)) + \
                  tf.reduce_sum(tf.abs(W_fc2)) + \
                  tf.reduce_sum(tf.abs(W_fc3)) + \
                  tf.reduce_sum(tf.abs(W_fc4)) + \
                  tf.reduce_sum(tf.abs(W_fc5)) +  \
                  tf.reduce_sum(tf.abs(W_fc6))
        loss += gamma * l1_loss
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
    return loss, y, gamma * l1_loss



def build_fc_freqSum_TiedWeight(x, x_dim, keep_prob, gamma=1e-7):
    # FC1
    with tf.name_scope("FC1"):
        #fc1_dim = 2048
        fc1_dim = 4096
        W_fc1 = weight_variable([x_dim, fc1_dim])
        b_fc1 = weight_variable([fc1_dim])
        h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)
        #h_fc1 = tf.matmul(x, W_fc1) + b_fc1

    # dropout
    with tf.name_scope("Dropout1"):
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # FC2
    with tf.name_scope("FC2"):
        #fc2_dim = 1024
        fc2_dim = 2048
        W_fc2 = weight_variable([fc1_dim, fc2_dim])
        b_fc2 = weight_variable([fc2_dim])
        #h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


    # FC3
    with tf.name_scope("FC3"):
        #fc3_dim = 512
        fc3_dim = 1024
        W_fc3 = weight_variable([fc2_dim, fc3_dim])
        b_fc3 = weight_variable([fc3_dim])
        #h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)
        h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)

    # FC4
    with tf.name_scope("FC4"):
        #fc4_dim = 256
        fc4_dim = 512
        W_fc4 = weight_variable([fc3_dim, fc4_dim])
        b_fc4 = weight_variable([fc4_dim])
        h_fc4 = tf.nn.relu(tf.matmul(h_fc3, W_fc4) + b_fc4)

    # FC5
    with tf.name_scope("FC5"):
        #fc5_dim = 128
        fc5_dim = 256
        W_fc5 = weight_variable([fc4_dim, fc5_dim])
        b_fc5 = weight_variable([fc5_dim])
        h_fc5 = tf.nn.relu(tf.matmul(h_fc4, W_fc5) + b_fc5)

    # FC6
    with tf.name_scope("FC6"):
        fc6_dim = 128 
        W_fc6 = weight_variable([fc5_dim, fc6_dim])
        b_fc6 = weight_variable([fc6_dim])
        h_fc6 = tf.nn.relu(tf.matmul(h_fc5, W_fc6) + b_fc6)

    # dropout
    #with tf.name_scope("Dropout2"):
    #    h_fc6_drop = tf.nn.dropout(h_fc6, keep_prob)

    # FC7
    with tf.name_scope("FC7"):
        #fc7_dim = 128
        fc7_dim = 256
        #W_fc7 = weight_variable([fc6_dim, fc7_dim])
        W_fc7 = tf.transpose(W_fc6)
        b_fc7 = weight_variable([fc7_dim])
        h_fc7 = tf.nn.relu(tf.matmul(h_fc6, W_fc7) + b_fc7)
        #h_fc7 = tf.nn.relu(tf.matmul(h_fc6_drop, W_fc7) + b_fc7)

    # FC8
    with tf.name_scope("FC8"):
        #fc8_dim = 256
        fc8_dim = 512
        #W_fc8 = weight_variable([fc7_dim, fc8_dim])
        W_fc8 = tf.transpose(W_fc5)
        b_fc8 = weight_variable([fc8_dim])
        h_fc8 = tf.nn.relu(tf.matmul(h_fc7, W_fc8) + b_fc8)

    # FC9
    with tf.name_scope("FC9"):
        #fc9_dim = 512
        fc9_dim = 1024
        #W_fc9 = weight_variable([fc8_dim, fc9_dim])
        W_fc9 = tf.transpose(W_fc4)
        b_fc9 = weight_variable([fc9_dim])
        h_fc9 = tf.nn.relu(tf.matmul(h_fc8, W_fc9) + b_fc9)


    # FC10
    with tf.name_scope("FC10"):
        #fc10_dim = 1024
        fc10_dim = 2048
        #W_fc10 = weight_variable([fc9_dim, fc10_dim])
        W_fc10 = tf.transpose(W_fc3)
        b_fc10 = weight_variable([fc10_dim])
        h_fc10 = tf.nn.relu(tf.matmul(h_fc9, W_fc10) + b_fc10)

    # dropout
    with tf.name_scope("Dropout3"):
        h_fc10_drop = tf.nn.dropout(h_fc10, keep_prob)

    # FC11
    with tf.name_scope("FC11"):
        #fc11_dim = 2048
        fc11_dim = 4096
        W_fc11 = tf.transpose(W_fc2)
        #W_fc11 = weight_variable([fc10_dim, fc11_dim])
        b_fc11 = weight_variable([fc11_dim])
        h_fc11 = tf.nn.relu(tf.matmul(h_fc10_drop, W_fc11) + b_fc11)
        #h_fc11 = tf.nn.relu(tf.matmul(h_fc10, W_fc11) + b_fc11)

    # FC12
    with tf.name_scope("FC12"):
        fc12_dim = x_dim
        #W_fc12 = weight_variable([fc11_dim, fc12_dim])
        W_fc12 = tf.transpose(W_fc1)
        b_fc12 = weight_variable([fc12_dim])
        #y = tf.nn.relu(tf.matmul(h_fc11, W_fc12) + b_fc12)
        y = tf.matmul(h_fc11, W_fc12) + b_fc12


    # LOSS 
    with tf.name_scope("loss"):
        #loss = tf.reduce_mean(tf.squared_difference(y, x))
        loss = tf.sqrt(tf.reduce_mean(tf.square(y-x)))
        l1_loss = tf.reduce_sum(tf.abs(W_fc1)) + \
                  tf.reduce_sum(tf.abs(W_fc2)) + \
                  tf.reduce_sum(tf.abs(W_fc3)) + \
                  tf.reduce_sum(tf.abs(W_fc4)) +  \
                  tf.reduce_sum(tf.abs(W_fc5)) +  \
                  tf.reduce_sum(tf.abs(W_fc6))
        #loss += gamma * l1_loss
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
    return loss, y, gamma * l1_loss






def build_fc_freqSum_TiedWeight_NoDropout(x, x_dim, keep_prob, gamma=1e-7):
    # FC1
    with tf.name_scope("FC1"):
        #fc1_dim = 2048
        fc1_dim = 4096
        W_fc1 = weight_variable([x_dim, fc1_dim])
        b_fc1 = weight_variable([fc1_dim])
        h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)
        #h_fc1 = tf.matmul(x, W_fc1) + b_fc1


    # FC2
    with tf.name_scope("FC2"):
        #fc2_dim = 1024
        fc2_dim = 2048
        W_fc2 = weight_variable([fc1_dim, fc2_dim])
        b_fc2 = weight_variable([fc2_dim])
        #h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    # dropout
    #with tf.name_scope("Dropout1"):
    #    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    # FC3
    with tf.name_scope("FC3"):
        #fc3_dim = 512
        fc3_dim = 1024
        W_fc3 = weight_variable([fc2_dim, fc3_dim])
        b_fc3 = weight_variable([fc3_dim])
        #h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)
        h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)

    # FC4
    with tf.name_scope("FC4"):
        #fc4_dim = 256
        fc4_dim = 512
        W_fc4 = weight_variable([fc3_dim, fc4_dim])
        b_fc4 = weight_variable([fc4_dim])
        h_fc4 = tf.nn.relu(tf.matmul(h_fc3, W_fc4) + b_fc4)

    # FC5
    with tf.name_scope("FC5"):
        #fc5_dim = 128
        fc5_dim = 256
        W_fc5 = weight_variable([fc4_dim, fc5_dim])
        b_fc5 = weight_variable([fc5_dim])
        h_fc5 = tf.nn.relu(tf.matmul(h_fc4, W_fc5) + b_fc5)

    # FC6
    with tf.name_scope("FC6"):
        fc6_dim = 128 
        W_fc6 = weight_variable([fc5_dim, fc6_dim])
        b_fc6 = weight_variable([fc6_dim])
        h_fc6 = tf.nn.relu(tf.matmul(h_fc5, W_fc6) + b_fc6)

    # dropout
    #with tf.name_scope("Dropout2"):
    #    h_fc6_drop = tf.nn.dropout(h_fc6, keep_prob)

    # FC7
    with tf.name_scope("FC7"):
        #fc7_dim = 128
        fc7_dim = 256
        #W_fc7 = weight_variable([fc6_dim, fc7_dim])
        W_fc7 = tf.transpose(W_fc6)
        b_fc7 = weight_variable([fc7_dim])
        h_fc7 = tf.nn.relu(tf.matmul(h_fc6, W_fc7) + b_fc7)
        #h_fc7 = tf.nn.relu(tf.matmul(h_fc6_drop, W_fc7) + b_fc7)

    # FC8
    with tf.name_scope("FC8"):
        #fc8_dim = 256
        fc8_dim = 512
        #W_fc8 = weight_variable([fc7_dim, fc8_dim])
        W_fc8 = tf.transpose(W_fc5)
        b_fc8 = weight_variable([fc8_dim])
        h_fc8 = tf.nn.relu(tf.matmul(h_fc7, W_fc8) + b_fc8)

    # FC9
    with tf.name_scope("FC9"):
        #fc9_dim = 512
        fc9_dim = 1024
        #W_fc9 = weight_variable([fc8_dim, fc9_dim])
        W_fc9 = tf.transpose(W_fc4)
        b_fc9 = weight_variable([fc9_dim])
        h_fc9 = tf.nn.relu(tf.matmul(h_fc8, W_fc9) + b_fc9)


    # FC10
    with tf.name_scope("FC10"):
        #fc10_dim = 1024
        fc10_dim = 2048
        #W_fc10 = weight_variable([fc9_dim, fc10_dim])
        W_fc10 = tf.transpose(W_fc3)
        b_fc10 = weight_variable([fc10_dim])
        h_fc10 = tf.nn.relu(tf.matmul(h_fc9, W_fc10) + b_fc10)

    # dropout
    #with tf.name_scope("Dropout3"):
    #    h_fc10_drop = tf.nn.dropout(h_fc10, keep_prob)

    # FC11
    with tf.name_scope("FC11"):
        #fc11_dim = 2048
        fc11_dim = 4096
        W_fc11 = tf.transpose(W_fc2)
        #W_fc11 = weight_variable([fc10_dim, fc11_dim])
        b_fc11 = weight_variable([fc11_dim])
        #h_fc11 = tf.nn.relu(tf.matmul(h_fc10_drop, W_fc11) + b_fc11)
        h_fc11 = tf.nn.relu(tf.matmul(h_fc10, W_fc11) + b_fc11)

    # FC12
    with tf.name_scope("FC12"):
        fc12_dim = x_dim
        #W_fc12 = weight_variable([fc11_dim, fc12_dim])
        W_fc12 = tf.transpose(W_fc1)
        b_fc12 = weight_variable([fc12_dim])
        #y = tf.nn.relu(tf.matmul(h_fc11, W_fc12) + b_fc12)
        y = tf.matmul(h_fc11, W_fc12) + b_fc12


    # LOSS 
    with tf.name_scope("loss"):
        #loss = tf.reduce_mean(tf.squared_difference(y, x))
        loss = tf.sqrt(tf.reduce_mean(tf.square(y-x)))
        l1_loss = tf.reduce_sum(tf.abs(W_fc1)) + \
                  tf.reduce_sum(tf.abs(W_fc2)) + \
                  tf.reduce_sum(tf.abs(W_fc3)) + \
                  tf.reduce_sum(tf.abs(W_fc4)) +  \
                  tf.reduce_sum(tf.abs(W_fc5)) +  \
                  tf.reduce_sum(tf.abs(W_fc6))
        #loss += gamma * l1_loss
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

    return loss, y, gamma * l1_loss




def build_fc_freqSum(x, x_dim, keep_prob, gamma=0.00001):
    # FC1
    with tf.name_scope("FC1"):
        fc1_dim = 2048
        #fc1_dim = 4096
        W_fc1 = weight_variable([x_dim, fc1_dim])
        b_fc1 = weight_variable([fc1_dim])
        h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)
        #h_fc1 = tf.matmul(x, W_fc1) + b_fc1

    # FC2
    with tf.name_scope("FC2"):
        fc2_dim = 1024
        #fc2_dim = 2048
        W_fc2 = weight_variable([fc1_dim, fc2_dim])
        b_fc2 = weight_variable([fc2_dim])
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    # dropout
    with tf.name_scope("Dropout1"):
        h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    # FC3
    with tf.name_scope("FC3"):
        fc3_dim = 512
        #fc3_dim = 1024
        W_fc3 = weight_variable([fc2_dim, fc3_dim])
        b_fc3 = weight_variable([fc3_dim])
        h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)
        #h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)

    # FC4
    with tf.name_scope("FC4"):
        fc4_dim = 256
        #fc4_dim = 512
        W_fc4 = weight_variable([fc3_dim, fc4_dim])
        b_fc4 = weight_variable([fc4_dim])
        h_fc4 = tf.nn.relu(tf.matmul(h_fc3, W_fc4) + b_fc4)

    # FC5
    with tf.name_scope("FC5"):
        fc5_dim = 128
        #fc5_dim = 256
        W_fc5 = weight_variable([fc4_dim, fc5_dim])
        b_fc5 = weight_variable([fc5_dim])
        h_fc5 = tf.nn.relu(tf.matmul(h_fc4, W_fc5) + b_fc5)

    # FC6
    with tf.name_scope("FC6"):
        fc6_dim = 128 
        W_fc6 = weight_variable([fc5_dim, fc6_dim])
        b_fc6 = weight_variable([fc6_dim])
        h_fc6 = tf.nn.relu(tf.matmul(h_fc5, W_fc6) + b_fc6)

    # dropout
    with tf.name_scope("Dropout2"):
        h_fc6_drop = tf.nn.dropout(h_fc6, keep_prob)

    # FC7
    with tf.name_scope("FC7"):
        fc7_dim = 128
        #fc7_dim = 256
        W_fc7 = weight_variable([fc6_dim, fc7_dim])
        b_fc7 = weight_variable([fc7_dim])
        #h_fc7 = tf.nn.relu(tf.matmul(h_fc6, W_fc7) + b_fc7)
        h_fc7 = tf.nn.relu(tf.matmul(h_fc6_drop, W_fc7) + b_fc7)

    # FC8
    with tf.name_scope("FC8"):
        fc8_dim = 256
        #fc8_dim = 512
        W_fc8 = weight_variable([fc7_dim, fc8_dim])
        b_fc8 = weight_variable([fc8_dim])
        h_fc8 = tf.nn.relu(tf.matmul(h_fc7, W_fc8) + b_fc8)

    # FC9
    with tf.name_scope("FC9"):
        fc9_dim = 512
        #fc9_dim = 1024
        W_fc9 = weight_variable([fc8_dim, fc9_dim])
        b_fc9 = weight_variable([fc9_dim])
        h_fc9 = tf.nn.relu(tf.matmul(h_fc8, W_fc9) + b_fc9)


    # FC10
    with tf.name_scope("FC10"):
        fc10_dim = 1024
        #fc10_dim = 2048
        W_fc10 = weight_variable([fc9_dim, fc10_dim])
        b_fc10 = weight_variable([fc10_dim])
        h_fc10 = tf.nn.relu(tf.matmul(h_fc9, W_fc10) + b_fc10)

    # dropout
    with tf.name_scope("Dropout3"):
        h_fc10_drop = tf.nn.dropout(h_fc10, keep_prob)

    # FC11
    with tf.name_scope("FC11"):
        fc11_dim = 2048
        #fc11_dim = 4096
        W_fc11 = weight_variable([fc10_dim, fc11_dim])
        b_fc11 = weight_variable([fc11_dim])
        h_fc11 = tf.nn.relu(tf.matmul(h_fc10_drop, W_fc11) + b_fc11)

    # FC12
    with tf.name_scope("FC12"):
        fc12_dim = x_dim
        W_fc12 = weight_variable([fc11_dim, fc12_dim])
        b_fc12 = weight_variable([fc12_dim])
        #y = tf.nn.relu(tf.matmul(h_fc11, W_fc12) + b_fc12)
        y = tf.matmul(h_fc11, W_fc12) + b_fc12


    # LOSS 
    with tf.name_scope("loss"):
        #loss = tf.reduce_mean(tf.squared_difference(y, x))
        loss = tf.sqrt(tf.reduce_mean(tf.square(y-x)))

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

    return loss, y



def build_fc_freqSum_L1(x, x_dim, keep_prob, gamma=0.00001):
    # FC1
    with tf.name_scope("FC1"):
        fc1_dim = 2048
        W_fc1 = weight_variable([x_dim, fc1_dim])
        b_fc1 = weight_variable([fc1_dim])
        h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

    # FC2
    with tf.name_scope("FC2"):
        fc2_dim = 1024
        W_fc2 = weight_variable([fc1_dim, fc2_dim])
        b_fc2 = weight_variable([fc2_dim])
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

     # dropout
    with tf.name_scope("Dropout1"):
        h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    # FC3
    with tf.name_scope("FC3"):
        fc3_dim = 512
        W_fc3 = weight_variable([fc2_dim, fc3_dim])
        b_fc3 = weight_variable([fc3_dim])
        h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

    # FC4
    with tf.name_scope("FC4"):
        fc4_dim = 256
        W_fc4 = weight_variable([fc3_dim, fc4_dim])
        b_fc4 = weight_variable([fc4_dim])
        h_fc4 = tf.nn.relu(tf.matmul(h_fc3, W_fc4) + b_fc4)

    # FC5
    with tf.name_scope("FC5"):
        fc5_dim = 128
        W_fc5 = weight_variable([fc4_dim, fc5_dim])
        b_fc5 = weight_variable([fc5_dim])
        h_fc5 = tf.nn.relu(tf.matmul(h_fc4, W_fc5) + b_fc5)

    # FC6
    with tf.name_scope("FC6"):
        fc6_dim = 128
        W_fc6 = weight_variable([fc5_dim, fc6_dim])
        b_fc6 = weight_variable([fc6_dim])
        h_fc6 = tf.nn.relu(tf.matmul(h_fc5, W_fc6) + b_fc6)

    # dropout
    with tf.name_scope("Dropout2"):
        h_fc6_drop = tf.nn.dropout(h_fc6, keep_prob)

    # FC7
    with tf.name_scope("FC7"):
        fc7_dim = 128
        W_fc7 = weight_variable([fc6_dim, fc7_dim])
        b_fc7 = weight_variable([fc7_dim])
        h_fc7 = tf.nn.relu(tf.matmul(h_fc6_drop, W_fc7) + b_fc7)

    # FC8
    with tf.name_scope("FC8"):
        fc8_dim = 256
        W_fc8 = weight_variable([fc7_dim, fc8_dim])
        b_fc8 = weight_variable([fc8_dim])
        h_fc8 = tf.nn.relu(tf.matmul(h_fc7, W_fc8) + b_fc8)

    # FC9
    with tf.name_scope("FC9"):
        fc9_dim = 512
        W_fc9 = weight_variable([fc8_dim, fc9_dim])
        b_fc9 = weight_variable([fc9_dim])
        h_fc9 = tf.nn.relu(tf.matmul(h_fc8, W_fc9) + b_fc9)

    # dropout
    with tf.name_scope("Dropout3"):
        h_fc9_drop = tf.nn.dropout(h_fc9, keep_prob)

    # FC10
    with tf.name_scope("FC10"):
        fc10_dim = 1024
        W_fc10 = weight_variable([fc9_dim, fc10_dim])
        b_fc10 = weight_variable([fc10_dim])
        h_fc10 = tf.nn.relu(tf.matmul(h_fc9_drop, W_fc10) + b_fc10)

    # FC11
    with tf.name_scope("FC11"):
        fc11_dim = 2048
        W_fc11 = weight_variable([fc10_dim, fc11_dim])
        b_fc11 = weight_variable([fc11_dim])
        h_fc11 = tf.nn.relu(tf.matmul(h_fc10, W_fc11) + b_fc11)

    # FC12
    with tf.name_scope("FC12"):
        fc12_dim = x_dim
        W_fc12 = weight_variable([fc11_dim, fc12_dim])
        b_fc12 = weight_variable([fc12_dim])
        y = tf.matmul(h_fc11, W_fc12) + b_fc12


    # LOSS 
    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.squared_difference(y, x))
        l1_loss = tf.reduce_sum(tf.abs(W_fc1)) + \
                  tf.reduce_sum(tf.abs(W_fc2)) + \
                  tf.reduce_sum(tf.abs(W_fc3)) + \
                  tf.reduce_sum(tf.abs(W_fc4)) +  \
                  tf.reduce_sum(tf.abs(W_fc5)) +  \
                  tf.reduce_sum(tf.abs(W_fc6)) + \
                  tf.reduce_sum(tf.abs(W_fc7)) + \
                  tf.reduce_sum(tf.abs(W_fc8)) +  \
                  tf.reduce_sum(tf.abs(W_fc9)) +  \
                  tf.reduce_sum(tf.abs(W_fc10)) + \
                  tf.reduce_sum(tf.abs(W_fc11)) + \
                  tf.reduce_sum(tf.abs(W_fc12)) 
        loss += gamma * l1_loss

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


    return loss, y
