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
import os
import argparse
import sys
from datetime import datetime
from pkg_resources import parse_version
from eeg_input_data import eeg_data
from eeg_org_data import eeg_subject_data
from utils import get_input_data_path, get_data_path_with_timestamp
from utils import build_model
from models.fc_freqSum_TiedWeight import build_fc_freqSum_TiedWeight
from models.fc_freqSum_TiedWeight import build_fc_freqSum_TiedWeight_NoBias
from models.fc_freqSum_TiedWeight import build_fc_freqSum_TiedWeight_NoDropout
from models.fc_freqSum_TiedWeight import build_fc_freqSum
from models.fc_freqSum_TiedWeight import build_fc_freqSum_TiedWeight_Big
from models.fc_freqSum_TiedWeight import build_fc_freqSum_NoTiedWeight_Big
from models.fc_freqSum_TiedWeight import build_fc_freqSum_NoTiedWeight_Small
from models.fc_freqSum_TiedWeight import build_fc_freqSum_NoTiedWeight_Medium
from models.fc_freqSum_TiedWeight import build_fc_freqSum_NoTiedWeight_Tiny
from models.fc_freqSum_TiedWeight import build_fc_freq_4_30_NoTiedWeight_Small
from models.fc_freqSum_TiedWeight import build_fc_freq_4_30_TiedWeight_Small
from models.fc_freqSum_TiedWeight import build_fc_freq_5_TiedWeight_Small
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

    subject_data = eeg._data
    subject_names = eeg._subjects
    X = subject_data[0]

    print('{} x {}'.format(X.shape[0], X.shape[1]))
    x_dim = X.shape[-1]

    # reset everything
    model_path = FLAGS.trained_model_base_dir

    # Input placeholder for input variables
    with tf.name_scope("input"):
        x = tf.placeholder(tf.float32, [None, x_dim])
        dropout_keep_prob = tf.placeholder(tf.float32)
        is_training = tf.placeholder(tf.bool, [])


    # BUILD MODEL
    gamma = 1e-7
    feature_activation = 'relu'

    loss, decoded, l1_loss = build_model(
        FLAGS.model,
        x,
        x_dim,
        dropout_keep_prob,
        gamma,
        feature_activation,
        is_training)

    # MODEL PATH
    model_path = FLAGS.trained_model_base_dir + '/' + FLAGS.model  + '/' + FLAGS.trained_model_name

    print("Model path: {}".format(model_path))
    meta_files = glob.glob(model_path + '/*.meta')
    meta_files.sort(key=os.path.getmtime)
    meta_file_fullpath = meta_files[-1]
    

    tf_version = tf.__version__.rpartition('.')[0]
    with tf.Session() as sess:
        tf.reset_default_graph()
        if parse_version(tf_version) >= parse_version('0.12.0'):
            sess.run(tf.global_variables_initializer())
        else:
            sess.run(tf.initialize_all_variables())
        # load the model
        saver = tf.train.import_meta_graph(meta_file_fullpath)
        check_point_dir = model_path + '/'
        saver.restore(sess, tf.train.latest_checkpoint(check_point_dir))

        for subject_idx, subject_name in  enumerate(subject_names):
            subject_data = subject_data[subject_idx]
            subject_time_features = []
            print("subject shape: {}".format(subject_data.shape))
            #for time_idx in range(0, subject_data.shape[0]):
                #batch_xs = subject_data[time_idx,:]
            batch_xs = subject_data
            feeds = {
                            x: batch_xs, 
                            dropout_keep_prob: 1.0, 
                            is_training: False
                    }
            subject_feature = sess.run('FC4/Elu', feed_dict=feeds)
            #subject_time_features.append(feature)
            #subject_features = np.vstack(subject_time_features)
            #print("{:6.5f}".format(subject_feature))
            #print("{} Feature shape: {}".format(subject_name, subject_feature.shape))

            #outFileName = subject_name + '.png'
            #plot_save_feature(subject_features, outFileName)


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