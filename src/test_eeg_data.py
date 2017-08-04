from eeg_input_data import eeg_data
from utils import get_input_data_path, get_data_path_with_timestamp
import numpy as np
import os
import glob
import h5py


#data_dir = '/home/chuong/EEG-Project/processed_data/volumes_freqSum'
#data_dir = '/home/chuong/volumes_freq_4_30/invcomp100Hz_sub30075_15remout3.mat_freq4_30_cortexsample1.h5'
data_dir = '/home/chuong/volumes_freq_4_30'

eeg = eeg_data()

normalization = 'normalize'
#eeg.get_data(data_dir, num_data_sec=-1,  fake=False, normalization='scaling')
eeg.get_data(data_dir, num_data_sec=-1,  fake=False,
        normalization=normalization)

data = eeg.images

