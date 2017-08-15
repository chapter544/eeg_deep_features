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


def plot_train_val_loss(data_path, fname):
    npz_fname = data_path + '/' + fname
    print("Reading {}".format(npz_fname))
    data = np.load(npz_fname)
    train = data['a']
    valid = data['b']
    n = np.arange(0, len(train))
    fig = plt.figure()
    plt.plot(n, train, '-b', label='train')
    plt.plot(n, valid, '-r', label='train')
    plt.ylabel('loss')
    plt.xlabel('iteration')
    plt.title(fname)
    fig.savefig(data_path + '/' + fname + '.png')
    plt.close(fig)



def main(_):
    ##################################################################
    # Read data 
    print("{} and {}".format(FLAGS.model, FLAGS.data_base_dir))

    # reset everything
    model_path = FLAGS.trained_model_base_dir

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

    print("{} and {}".format(model_path, FLAGS.model))
    # plot train/val loss
    try:
        plot_train_val_loss(model_path, FLAGS.model + '_epoch_.npz')
    except:
        pass


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
