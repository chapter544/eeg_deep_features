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
from eeg_input_data import eeg_data
from utils import get_input_data_path, get_data_path

# Read data 
model = 'freqSumBig'
data_base_dir = '/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/processed_data'
sub_volumes_dir = get_input_data_path(model, data_base_dir)

eeg = eeg_data()
eeg.get_data(sub_volumes_dir, num_data_sec=-1,  fake=False)
data = eeg.images

