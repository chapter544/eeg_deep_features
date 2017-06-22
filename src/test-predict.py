# Auto-encoder for 2D cortex slices
# Chuong Nguyen
# 07/11/2017
from __future__ import print_function

import tensorflow as tf
import numpy as np
import h5py
import glob
import os
import pickle
from datetime import datetime
import argparse
from utils import get_input_data_path, get_data_path



def matstat(x):
    maxVal = np.max(x)
    minVal = np.min(x)
    stdVal = np.std(x)
    meanVal = np.mean(x)
    medianVal = np.median(x)
    print("Max: {}, Min: {}, Mean: {}".format(maxVal, minVal, meanVal))
    print("std: {}, median: {}".format(stdVal, meanVal))

def numzeros(x, T):
    row, col = x.shape
    return 100.0 * np.sum(np.abs(x) < T) / (row*col) 


model = 'freqSumBig'
# L1 model with gamma = 1e-7
#trained_model_name = '2017-06-20-095105'
#meta_file = 'freqSumBig_epoch_500_2017-06-20-142652.ckpt-78500.meta'

# Normal model without L1
#trained_model_name = '2017-06-20-072834'
#meta_file = 'freqSumBig_epoch_120_2017-06-20-083437.ckpt-18840.meta'

trained_model_base_dir = '/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/trained_models'
model_path = trained_model_base_dir + '/' + model  + '/' + trained_model_name


# start tensorflow session
sess = tf.Session()

# load network and weights
meta_file_fullpath = model_path + '/' + meta_file
saver = tf.train.import_meta_graph(meta_file_fullpath)

# checkpoint location
check_point_dir = model_path + '/'
saver.restore(sess, tf.train.latest_checkpoint(check_point_dir))
print("Done restoring model ...")


#########################################################################
# GET VARIABLES 
##########################################################################
# set default graph
graph = tf.get_default_graph()

# get variables using scope FC1
W_fc1 = tf.get_collection(tf.GraphKeys.VARIABLES, scope='FC1')[0]
b_fc1 = tf.get_collection(tf.GraphKeys.VARIABLES, scope='FC1')[1]

# get variables using scope FC2


trained_model_base_dir = '/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/trained_models'
model_path = trained_model_base_dir + '/' + model  + '/' + trained_model_name


# start tensorflow session
sess = tf.Session()

# load network and weights
meta_file_fullpath = model_path + '/' + meta_file
saver = tf.train.import_meta_graph(meta_file_fullpath)

# checkpoint location
check_point_dir = model_path + '/'
saver.restore(sess, tf.train.latest_checkpoint(check_point_dir))
print("Done restoring model ...")


#########################################################################
# GET VARIABLES 
##########################################################################
# set default graph
graph = tf.get_default_graph()

# get variables using scope FC1
W_fc1 = tf.get_collection(tf.GraphKeys.VARIABLES, scope='FC1')[0]
b_fc1 = tf.get_collection(tf.GraphKeys.VARIABLES, scope='FC1')[1]

# get variables using scope FC2
W_fc2 = tf.get_collection(tf.GraphKeys.VARIABLES, scope='FC2')[0]
b_fc2 = tf.get_collection(tf.GraphKeys.VARIABLES, scope='FC2')[1]

# get variables using scope FC3
W_fc3 = tf.get_collection(tf.GraphKeys.VARIABLES, scope='FC3')[0]
b_fc3 = tf.get_collection(tf.GraphKeys.VARIABLES, scope='FC3')[1]

# get variables using scope FC4
W_fc4 = tf.get_collection(tf.GraphKeys.VARIABLES, scope='FC4')[0]
b_fc4 = tf.get_collection(tf.GraphKeys.VARIABLES, scope='FC4')[1]

# get variables using scope FC5
W_fc5 = tf.get_collection(tf.GraphKeys.VARIABLES, scope='FC5')[0]
b_fc5 = tf.get_collection(tf.GraphKeys.VARIABLES, scope='FC5')[1]

# get variables using scope FC6
W_fc6 = tf.get_collection(tf.GraphKeys.VARIABLES, scope='FC6')[0]
b_fc6 = tf.get_collection(tf.GraphKeys.VARIABLES, scope='FC6')[1]

[w_fc1, b1, w_fc2, b2, w_fc3, b3, w_fc4, b4, w_fc5, b5, w_fc6, b6] = sess.run([W_fc1, b_fc1, W_fc2, b_fc2, W_fc3, b_fc3, W_fc4, b_fc4, W_fc5, b_fc5, W_fc6, b_fc6])

sess.close()
