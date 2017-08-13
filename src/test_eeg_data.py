from eeg_input_data import eeg_data
from utils import get_input_data_path, get_data_path_with_timestamp
import numpy as np
import os
import glob
import h5py
import matplotlib.pyplot as plt


#data_dir = '/home/chuong/EEG-Project/processed_data/volumes_freqSum'
#data_dir = '/home/chuong/volumes_freq_4_30/invcomp100Hz_sub30075_15remout3.mat_freq4_30_cortexsample1.h5'
#data_dir = '/home/chuong/volumes_freq_4_30'
data_dir = '/home/chuong/volumes_freq_5'
num_val_samples = 1000

eeg = eeg_data()

normalization = 'normalize'
#eeg.get_data(data_dir, num_data_sec=-1,  fake=False, normalization='scaling')
eeg.get_data(data_dir, num_data_sec=-1,  fake=False, normalization=normalization)
#train_data, valid_data = eeg.get_data_from_files(data_dir, num_val_samples, num_data_sec=-1, fake=False)

train_data = eeg._train_data
valid_data = eeg._valid_data
