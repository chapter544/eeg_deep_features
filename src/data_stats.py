# Auto-encoder for 2D cortex slices
# Chuong Nguyen
# 07/11/2017
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import h5py
import glob
import os
import argparse
import sys
from datetime import datetime
from eeg_input_data import eeg_data
from utils import get_input_data_path, get_data_path_with_timestamp
import matplotlib.pyplot as plt
import pickle


data_models = [
        'freqSum_NoTiedWeight_BN_Wrapper_Behind_Tiny',
        'freq_5_NoTiedWeight_BN_Wrapper_Behind_Tiny',
        'freq_4_30_NoTiedWeight_BN_Wrapper_Behind_Tiny']

data_base_dir = [
        '/home/chuong/EEG-Project/processed_data',
        '/home/chuong/volumes_freq_5',
        '/home/chuong/volumes_freq_4_30']

for subject_idx, model in enumerate(data_models):
    print("Doing {}".format(model))
    sub_volumes_dir = get_input_data_path(model, data_base_dir[subject_idx])
    eeg = eeg_data()
    data, subject_names = eeg.get_data_list_from_files(sub_volumes_dir)
    subject_stats = []
    for i in range(0, len(data)):
        data_mean = np.mean(data[i])
        data_max = np.max(data[i])
        data_min = np.min(data[i])
        data_std = np.std(data[i])
        data_median = np.median(data[i])
        data_stats = np.array([data_max, data_min, data_mean, data_median,
            data_std])
        subject_stats.append(data_stats)

    subject_stats_np = np.vstack(subject_stats)
    fig = plt.figure()
    plt.plot(subject_stats_np[:,0])
    plt.title('Max')
    plt.savefig(model + '_max.png')
    plt.close()

    fig = plt.figure()
    plt.plot(subject_stats_np[:,1])
    plt.title('Min')
    plt.savefig(model + '_min.png')
    plt.close()

    fig = plt.figure()
    plt.plot(subject_stats_np[:,2])
    plt.title('Mean')
    plt.savefig(model + '_mean.png')
    plt.close()

    fig = plt.figure()
    plt.plot(subject_stats_np[:,3])
    plt.title('Median')
    plt.savefig(model + '_median.png')
    plt.close()

    with open(model + '.pickle', 'wb') as f: 
        pickle.dump([subject_names, subject_stats_np], f)

#with open(model + '.pickle', 'r') as f: 
#    names, stats = pickle.load(f)
