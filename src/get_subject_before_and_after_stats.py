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
import pickle


data_models = [
        'freqSum_NoTiedWeight_BN_Wrapper_Behind_Tiny',
        'freq_5_NoTiedWeight_BN_Wrapper_Behind_Tiny',
        'freq_4_30_NoTiedWeight_BN_Wrapper_Behind_Tiny']

data_runs = [
        '2017-08-16-160815-elu',
        '2017-08-16-165112-elu',
        '2017-08-15-184256-relu']

data_base_dir = [
        '/home/chuong/EEG-Project/processed_data',
        '/home/chuong/volumes_freq_5',
        '/home/chuong/volumes_freq_4_30']

output_feature_dir = '/home/chuong/EEG-Project/output_features'

for subject_idx, model in enumerate(data_models):
    feature_fname = output_feature_dir  + '/' + model + '/' + data_runs[subject_idx] + '/' + 'deep_feature.pkl'
    print("Feature fname: {}".format(feature_fname))

    #with open(model + '.pickle', 'wb') as f: 
    #    pickle.dump([subject_names, subject_stats_np], f)

    # input data information
    with open(model + '.pickle', 'r') as f: 
        names, stats = pickle.load(f)

    # output feature information
    with open(feature_fname, 'r') as f:
        sub_names, features = pickle.load(f)
        print("Len {}".format(len(sub_names)))

    for a, b in zip(names, sub_names):
        if a != b:
            print("Invalid: {} {}".format(a,b))

    out_data = []
    for sub_idx in range(0, len(sub_names)):
        before_stats = stats[sub_idx]
        after_data = features[sub_idx]
        out_stats = [before_stats[0], before_stats[1], np.max(after_data),
                np.min(after_data)]
        out_data.append(out_stats)

    with open(model + '_before_after_max_min_stats.txt', 'w') as f: 
        for sub_idx in range(0, len(sub_names)):
            line_data = out_data[sub_idx]
            line = "{} & {} & {} & {} & {}\\\\\n".format(sub_names[sub_idx], line_data[0], line_data[2], line_data[1], line_data[3])
            f.write(line)
