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
from utils import get_input_data_path, get_data_path_with_timestamp

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, 
        default='big', 
        help='Model architecture (big, small, freqSumSmall, freqSumBig)')
parser.add_argument('--trained_model_name', type=str, 
        default='', 
        help='Trained specific model directory')
parser.add_argument('--data_normalization', type=str, 
        default='normalize', 
        help='Type of data normalization (normalize, scaling)')
parser.add_argument('--feature_activation', type=str, 
        default='linear', 
        help='Activation function for the feature layer(linear, relu)')
parser.add_argument('--data_base_dir', type=str, 
        default='/home/chuong/EEG-Project/processed_data', 
        help='Data base directory')
parser.add_argument('--trained_model_base_dir', type=str, 
        default='/home/chuong/EEG-Project/trained_models', 
        help='Trained model directory')
parser.add_argument('--output_base_dir', type=str, 
        default='/home/chuong/EEG-Project/output_features', 
        help='Output directory')
FLAGS, unparsed = parser.parse_known_args()


if FLAGS.trained_model_name == '':
    raise Exception("A trained model name is required (--trained_model_name). It is usually a timestamp string")

if FLAGS.output_base_dir == '':
    raise Exception("Model result directory is required")

# sub_volumes directory, 
sub_volumes_dir = get_input_data_path(FLAGS.model, FLAGS.data_base_dir)

# feature activation (linear, relu)
feature_activation = FLAGS.feature_activation

# DEFINE model to compute
model_name = FLAGS.model
output_dir = FLAGS.output_base_dir+ '/' + model_name + '-' + FLAGS.trained_model_name
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def read_hdf5(fName):
    f = h5py.File(fName, 'r')
    data = np.array(f['data'])
    return data


def get_feature_by_subject(data_dir, W, b, 
                           normalization='normalize', activation='relu'):
    data = dict()
    os.chdir(data_dir)

    for fName in glob.glob("*.h5"):
        fNameFullPath = data_dir + '/' + fName
        print('  Working on {}'.format(fName))
        eeg_subject_data = read_hdf5(fNameFullPath)
        time_size = eeg_subject_data.shape[0]

        if model == "big" or model == "small":
            data_time_space = eeg_subject_data.reshape(time_size,-1)
        elif model == "freqSumSmall" or model == "freqSumBig":
            data_time_space = eeg_subject_data
        else:
            data_time_space = eeg_subject_data

        if normalization == 'normalize':
            print("Normalizing data ...")
            normalized_data = np.load('normalization.npz')
            mean_val = normalized_data['mean_val']
            std_val = normalized_data['std_val']
            data_time_space = (data_time_space - mean_val) / std_val
        elif normalization == 'scaling':
            print("Scaling data data ...")
            normalized_data = np.load('scaling.npz')
            max_val = normalized_data['max_val']
            min_val = normalized_data['min_val']
            data_time_space = (2.0 * (data_time_space - min_val) / 
                              (max_val - min_val)) - 1.0

        # iterate and multiply all of them
        if len(b) == 0:
            data_time_feature = data_time_space.dot(W[0])
            for i in range(1, len(W)-1):
                data_time_feature = data_time_feature.dot(W[i])

            last_layer_idx = len(W)-1
            temp_data = data_time_feature.dot(W[last_layer_idx])
            if feature_activation == 'linear':
                data_time_feature = temp_data 
            else:
                data_time_feature = np.maximum(temp_data, 0, temp_data)
        elif len(b) == len(W):
            temp_data = data_time_space.dot(W[0]) + b[0]
            data_time_feature = np.maximum(temp_data, 0, temp_data)
            for i in range(1, len(W)-1):
                temp_data = data_time_feature.dot(W[i]) + b[i]
                data_time_feature = np.maximum(temp_data, 0, temp_data)
            last_layer_idx = len(W)-1
            temp_data = data_time_feature.dot(W[last_layer_idx]) + \
                        b[last_layer_idx]
            if feature_activation == 'linear':
                data_time_feature = temp_data 
            else:
                data_time_feature = np.maximum(temp_data, 0, temp_data)


        #data_time_feature_2 = data_time_feature_1.dot(W['w2']) + b['b2']
        #data_time_feature_3 = data_time_feature_2.dot(W['w3']) + b['b3']
        #data_time_feature_4 = data_time_feature_3.dot(W['w4']) + b['b4']
        #data_time_feature_5 = data_time_feature_4.dot(W['w5']) + b['b5']
        #data_time_feature_6 = data_time_feature_5.dot(W['w6']) + b['b6']
        #data_time_feature = data_time_feature_6

        data[fName] = data_time_feature.transpose()
    return data


model = FLAGS.model

print("{0}, {1}, {2}".format(FLAGS.trained_model_base_dir, FLAGS.model, FLAGS.trained_model_name))
model_path = FLAGS.trained_model_base_dir + '/' + FLAGS.model  + '/' + FLAGS.trained_model_name


# load network and weights

meta_files = glob.glob(model_path + '/*.meta')
meta_files.sort(key=os.path.getmtime)
meta_file_fullpath = meta_files[-1]

# start tensorflow session
sess = tf.Session()
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

# one can use tf.get_collection to retrieve the variable names
# this name matches the tensorboard variable also.
# for i in tf.get_collection(tf.GraphKeys.VARIABLES, scope='FC1'):
#   print i.name
if model == "small" or model == "big" or model == 'freqSum_NoTiedWeight_Small':
    # get variables using scope FC1
    W_fc1 = tf.get_collection(tf.GraphKeys.VARIABLES, scope='FC1')[0]
    b_fc1 = tf.get_collection(tf.GraphKeys.VARIABLES, scope='FC1')[1]

    # get variables using scope FC2
    W_fc2 = tf.get_collection(tf.GraphKeys.VARIABLES, scope='FC2')[0]
    b_fc2 = tf.get_collection(tf.GraphKeys.VARIABLES, scope='FC2')[1]

    # get variables using scope FC2
    W_fc3 = tf.get_collection(tf.GraphKeys.VARIABLES, scope='FC3')[0]
    b_fc3 = tf.get_collection(tf.GraphKeys.VARIABLES, scope='FC3')[1]

    [w_fc1, b1, w_fc2, b2, w_fc3, b3] = sess.run([W_fc1, b_fc1, W_fc2, b_fc2, W_fc3, b_fc3])

    W = []
    W.append(w_fc1)
    W.append(w_fc2)
    W.append(w_fc3)

    b = []
    b.append(b1)
    b.append(b2)
    b.append(b3)
elif model == "freqSumSmall" or model == "freqSumBig" or model == 'freqSum_TiedWeight_Big' or model == 'freqSum_TiedWeight' or model == 'freqSum_NoTiedWeight_Medium':
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
    [w_fc1, b1, w_fc2, b2, w_fc3, b3, w_fc4, b4] = sess.run([W_fc1, b_fc1, W_fc2, b_fc2, W_fc3, b_fc3, W_fc4, b_fc4])

    W = []
    W.append(w_fc1)
    W.append(w_fc2)
    W.append(w_fc3)
    W.append(w_fc4)

    b = []
    b.append(b1)
    b.append(b2)
    b.append(b3)
    b.append(b4)

elif model == "freqSum_NoTiedWeight_Big":
    print("Doing model %s " % (model))
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

    W = []
    W.append(w_fc1)
    W.append(w_fc2)
    W.append(w_fc3)
    W.append(w_fc4)
    W.append(w_fc5)
    W.append(w_fc6)

    b = []
    b.append(b1)
    b.append(b2)
    b.append(b3)
    b.append(b4)
    b.append(b5)
    b.append(b6)


# close tensorflow session
sess.close()
print("Obtaining transformation matrix W, and closing model ...")

# get the time_feature from each subject,
# for now, it's either using num_data_sec 
# or we can compute the whole volume
# construct the transformation matrix
#data_no_bias = get_feature_by_subject(sub_volumes_dir, W, [])

#dump_file_no_bias = output_dir + '/' + 'volumes_time_feature_no_bias.pkl'
#with open(dump_file_no_bias, 'wb') as output:
#    pickle.dump(data_no_bias, output)
#    pickle.dump(W, output)
#

data_with_bias = get_feature_by_subject(sub_volumes_dir, W, b)

dump_file_with_bias = output_dir + '/' + 'volumes_time_feature_with_bias.pkl'
with open(dump_file_with_bias, 'wb') as output:
    pickle.dump(data_with_bias, output)
    pickle.dump(W, output)
    pickle.dump(b, output)
