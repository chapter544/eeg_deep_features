import numpy as np
import h5py
import glob
import os

class eeg_data(object):
    def __init__(self):
        self._train_data = None
        self._labels = None
        self._valid_data = None
        self._num_examples = 0
        self._epochs_completed = 0
        self._index_in_epoch = 0


    def get_data_from_files(self, data_dir, 
            num_val_samples, num_data_sec=180,
            fake=False ):
        data = []
        if fake == False:
            os.chdir(data_dir)
            for fName in glob.glob("*.h5"):
                fNameFullPath = data_dir + '/' + fName
                eeg_subject_data = self.read_hdf5(fNameFullPath)
                if num_data_sec == -1:
                    data.append(eeg_subject_data)
                else:
                    data.append(eeg_subject_data.reshape(num_data_sec,-1))
        else: # fake data generation
            if num_data_sec == -1:
                data.append(np.random.randn(720, 18715))
            else:
                data.append(np.random.randn(720, 250000))

        # group all data together
        all_data = np.vstack(data)
        np.random.shuffle(all_data)
        #num_val_samples = 500
        val_data = all_data[:num_val_samples]
        train_data = all_data[num_val_samples:]

        return train_data, val_data



    def get_data(self, data_dir, num_data_sec=180, 
            fake=False, normalization='normalize'):
        num_val_samples = 1000
        train_data, val_data =  self.get_data_from_files(data_dir, 
                num_val_samples, num_data_sec, fake)

        if normalization == 'scaling':

            max_val = np.max(train_data)
            min_val = np.min(train_data)
            print("Scaling features ....")
            np.savez('scaling.npz', max_val=max_val, min_val=min_val)
            self._valid_data = val_data
            self._train_data = train_data
            self._train_data = np.divide(
                                         (self._train_data - min_val),
                                         (max_val - min_val)
                                        ) * 2.0  - 1.0 
            self._validation = np.divide(
                                         (self._valid_data - min_val), 
                                         (max_val - min_val)
                                        ) * 2.0 - 1.0
        elif normalization == 'normalize':
            print("Normalizing features ....")
            mean_data = np.mean(train_data, axis=0)
            std_data = np.std(train_data, axis=0)
            np.savez('normalization.npz', mean_val=mean_data, std_val=std_data)
            train_data -= mean_data
            train_data /=  std_data
            self._train_data = train_data
            self._valid_data = (val_data - mean_data) / std_data
        else:
            self._valid_data = valid_data
            self._train_data = train_data

        # quick test overfitting
        #self._data = self._data[:100]

        self._labels = None
        self._num_examples = self._train_data.shape[0]
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def epochs_completed(self):
        return self._epochs_completed

    @property
    def train_data(self):
        return self._train_data

    @property
    def valid_data(self):
        return self._valid_data

    @property
    def labels(self):
        return self._labels

    def read_hdf5(self, fName):
        f = h5py.File(fName, 'r')
        data = np.array(f['data'])
        return data


    def iterate_minibatches(self, batchsize, shuffle=True):
        if shuffle:
            indices = np.arange(self._train_data.shape[0])
            np.random.shuffle(indices)
        for start_idx in range(0, self._train_data.shape[0] - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield self._train_data[excerpt]


    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._datac = self._train_data[perm0]
            #self._labels = self._labels[perm0]

        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            images_rest_part = self._train_data[start:self._num_examples]
            labels_rest_part = None

            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._data = self._train_data[perm]
                #self._labels = self._labels[perm]

            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._train_data[start:end]
            #labels_new_part = self._labels[start:end]
            labels_new_part = None
            return np.concatenate((images_rest_part, images_new_part), axis=0), None
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._train_data[start:end], None

