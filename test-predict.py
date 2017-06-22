import tensorflow as tf
import numpy as np
import h5py
import glob
import os
import pickle
from datetime import datetime
import argparse
from utils import get_input_data_path, get_data_path
import matplotlib.pyplot as plt

model = 'freqSumBig'
trained_model_name='2017-06-20-095105'
data_base_dir='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/processed_data'
output_base_dir='/home/chuong/EEG-Project/output_features'
sub_volumes_dir = get_input_data_path(model, data_base_dir)


def read_hdf5(fName):
    f = h5py.File(fName, 'r')
    data = np.array(f['data'])
    return data


trained_model_base_dir = '/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/trained_models'

model_path = trained_model_base_dir + '/' + model  + '/' + trained_model_name


meta_file_fullpath = model_path + '/' + 'freqSumBig_epoch_500_2017-06-20-142652.ckpt-78500.meta'

# start tensorflow session
sess = tf.Session()

# load network and weights
saver = tf.train.import_meta_graph(meta_file_fullpath)

# checkpoint location
check_point_dir = model_path + '/'
saver.restore(sess, tf.train.latest_checkpoint(check_point_dir))
print("Done restoring model ...")


W_fc1 = tf.get_collection(tf.GraphKeys.VARIABLES, scope='FC1')[0]
b_fc1 = tf.get_collection(tf.GraphKeys.VARIABLES, scope='FC1')[1]

# get variables using scope FC2
W_fc2 = tf.get_collection(tf.GraphKeys.VARIABLES, scope='FC2')[0]
b_fc2 = tf.get_collection(tf.GraphKeys.VARIABLES, scope='FC2')[1]

# get variables using scope FC2
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


# close tensorflow session
sess.close()
print("Obtaining transformation matrix W, and closing model ...")


fName = 'invcomp100Hz_sub30225_9remout3.mat_freqSum.h5'
data_dir = '/home/chuong/EEG-Project/processed_data/volumes_freqSum'
fNameFullPath = data_dir + '/' + fName
print('  Working on {}'.format(fName))
eeg_subject_data = read_hdf5(fNameFullPath)


v1 = eeg_subject_data.dot(w_fc1) + b1
v1 = v1 * (v1 > 0)

v2 = v1.dot(w_fc2) + b2
v2 = v2 * (v2 > 0)

v3 = v2.dot(w_fc3) + b3
v3 = v3 * (v3 > 0)

v4 = v3.dot(w_fc4) + b4
v4 = v4 * (v4 > 0)

v5 = v4.dot(w_fc5) + b5
v5 = v5 * (v5 > 0)

v6 = v5.dot(w_fc6) + b6
v6 = v6 * (v6 > 0)

showMat(v6.transpose())






def showMat(x):
	plt.imshow(x)
	plt.colorbar()
	plt.show()
