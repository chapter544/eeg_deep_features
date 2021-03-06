# Auto-encoder for 2D cortex slices
# Chuong Nguyen
# 07/11/2017
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import h5py
import glob
import pickle
import os
import argparse
import sys
from datetime import datetime
from pkg_resources import parse_version
from eeg_input_data import eeg_data
from eeg_org_data import eeg_subject_data
from utils import get_input_data_path, get_data_path_with_timestamp
import models


import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt 


FLAGS = None


def plot_feature(data, fName):
    plt.imshow(data, aspect='auto') 
    plt.colorbar() 
    plt.tight_layout() 
    plt.savefig(fName)
    plt.close() 


def plot_train_val_loss(data_path, fname):
    data_fname = data_path + '/' + fname
    print("Loading data {}".format(data_fname))
    data = np.load(data_fname)
    train = data['a']
    valid = data['b']
    n = np.arange(0, len(train))
    fig = plt.figure()
    plt.plot(n, train, '-b', label='train')
    plt.plot(n, valid, '-r', label='train')
    plt.ylabel('loss')
    plt.xlabel('iteration')
    plt.title(fname)
    fig.savefig(data_fname + '.png')
    plt.close(fig)



def main(_):
    ##################################################################
    # Read data 
    print("{} and {}".format(FLAGS.model, FLAGS.data_base_dir))
    sub_volumes_dir = get_input_data_path(FLAGS.model, FLAGS.data_base_dir)
    eeg = eeg_subject_data()

    if FLAGS.data_type == 'subsample': # subsample on 3D axes
        eeg.get_data(sub_volumes_dir, fake=FLAGS.test)
    else: # no subsampling
        eeg.get_data(sub_volumes_dir, num_data_sec=-1, 
                fake=False, normalization=FLAGS.data_normalization)

    subjects_data = eeg._data
    subject_names = eeg._subjects
    X = subjects_data[0]

    print('{} x {}'.format(X.shape[0], X.shape[1]))
    x_dim = X.shape[-1]

    # reset everything
    model_path = FLAGS.trained_model_base_dir

    # Input placeholder for input variables
    #with tf.name_scope("input"):
    #    x = tf.placeholder(tf.float32, [None, x_dim])
    #    dropout_keep_prob = tf.placeholder(tf.float32)
    #    is_training = tf.placeholder(tf.bool, [])

    # create output directory for image features
    output_base_dir = FLAGS.output_base_dir
    output_model_dir = output_base_dir + '/' + FLAGS.model
    if not os.path.exists(output_model_dir):
        os.makedirs(output_model_dir)
    output_dir = output_model_dir + '/' + FLAGS.trained_model_name
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    # MODEL PATH
    model_path = FLAGS.trained_model_base_dir + '/' + FLAGS.model  + '/' + FLAGS.trained_model_name

    print("Model path: {}".format(model_path))
    meta_files = glob.glob(model_path + '/*.meta')
    meta_files.sort(key=os.path.getmtime)
    meta_file_fullpath = meta_files[-1]

    # plot train/val loss
    try:
        plot_train_val_loss(model_path, FLAGS.model + '_epoch_.npz')
    except:
        pass

    tf_version = tf.__version__.rpartition('.')[0]
    with tf.Session() as sess:
        #tf.reset_default_graph()
        # load the model
        saver = tf.train.import_meta_graph(meta_file_fullpath)
        check_point_dir = model_path + '/'
        saver.restore(sess, tf.train.latest_checkpoint(check_point_dir))

        graph = tf.get_default_graph()
        try:
            x = graph.get_tensor_by_name('input/Placeholder:0')
            dropout_keep_prob = graph.get_tensor_by_name('input/Placeholder_1:0')
            is_training = graph.get_tensor_by_name('input/Placeholder_2:0')
            feature_op = graph.get_tensor_by_name('FCFeat/feature:0')
        except:
            x = graph.get_tensor_by_name('x:0')
            dropout_keep_prob = graph.get_tensor_by_name('training_phase:0')
            is_training = graph.get_tensor_by_name('dropout_rate:0')
            feature_op = graph.get_tensor_by_name('FCFeat/FCFeat:0')

        output_features = [] 
        for subject_idx, subject_name in  enumerate(subject_names):
            sub_data = subjects_data[subject_idx]
            subject_time_features = []
            print("subject shape: {}".format(sub_data.shape))
            batch_xs = sub_data
            feeds = {
                            x: batch_xs, 
                            dropout_keep_prob: 1.0, 
                            is_training: False
                    }
            sub_feature = sess.run(feature_op, feed_dict=feeds)
            output_features.append(sub_feature)


    for subject_idx, subject_name in enumerate(subject_names):
        subject_feat = output_features[subject_idx]
        plot_feature(subject_feat.transpose(), 
                     output_dir + '/' + subject_name + '.png')

    # save it out to a file
    with open(output_dir + '/deep_feature.pkl', 'wb') as f:
        pickle.dump([subject_names, output_features], f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, 
                default='big', 
                    help='Architecture (big, small, freqSumSmall, freqSumBig)')
    parser.add_argument('--data_type', type=str, 
             default='subsample', 
             help='Subsampling (subsample, freqSum)')
    parser.add_argument('--model_name', type=str, 
                default='', 
                    help='Trained model name')
    parser.add_argument('--data_base_dir', type=str, 
                default='', 
                    help='Data base directory')
    parser.add_argument('--trained_model_base_dir', type=str, 
                    default='/home/chuong/EEG-Project/trained_models', 
                            help='Trained model directory')
    parser.add_argument('--output_base_dir', type=str, 
                default='/home/chuong/EEG-Project/output_features', 
                    help='Output base directory')
    parser.add_argument('--data_normalization', type=str, 
                default='scaling', 
                help='Data normalization: scaling, normalize, none')
    parser.add_argument('--trained_model_name', type=str, 
                default='', 
                            help='Trained specific model directory')


    FLAGS, unparsed = parser.parse_known_args()
    #tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    tf.app.run(main=main)
