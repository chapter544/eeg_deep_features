from eeg_input_data import eeg_data
from utils import get_input_data_path, get_data_path_with_timestamp
import numpy as np
import os
import glob
import h5py


def read_hdf5(fName):
	f = h5py.File(fName, 'r')
	data = np.array(f['data'])
	return data




data_dir = '/home/chuong/EEG-Project/processed_data/volumes_freqSum'

data = []
os.chdir(data_dir)
for fName in glob.glob("*.h5"):
	fNameFullPath = data_dir + '/' + fName
	eeg_subject_data = read_hdf5(fNameFullPath)
	data.append(eeg_subject_data)


all_data = np.vstack(data)
