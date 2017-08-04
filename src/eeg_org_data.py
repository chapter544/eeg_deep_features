import numpy as np
import h5py
import glob
import os

class eeg_subject_data(object):
    def __init__(self):
        self._data = []
        self._subjects = []

    def get_data(self, data_dir, num_data_sec=180, fake=False, normalization='normalize'): 
		data = []

		# get z-transform data
		if normalization == 'normalize':
			normalized_data = np.load(data_dir + '/' + 'normalization.npz')
			mean_val = normalized_data['mean_val']
			std_val = normalized_data['std_val']

		# get min-max scaling data
		if normalization == 'scaling':
			normalized_data = np.load(data_dir + '/' + 'scaling.npz')
			max_val = normalized_data['max_val']
			min_val = normalized_data['min_val']

		if fake == False:
			os.chdir(data_dir)
			for fName in glob.glob("*.h5"):
				fNameFullPath = data_dir + '/' + fName
				data_time_space = self.read_hdf5(fNameFullPath)
				if normalization == 'normalize':
					print("Normalizing data ...")
					data_time_space = (data_time_space - mean_val) / std_val
				elif normalization == 'scaling':
					data_time_space = (2.0 * (data_time_space - min_val) / (max_val - min_val)) - 1.0

				self._data.append(data_time_space)
				self._subjects.append(os.path.basename(fName))


    def read_hdf5(self, fName):
        f = h5py.File(fName, 'r')
        data = np.array(f['data'])
        return data

