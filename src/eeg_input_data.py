import numpy as np
import h5py
import glob
import os

class eeg_data(object):
    def __init__(self):
        self._images = None
        self._labels = None
        self._test = None
        self._num_examples = 0
        self._epochs_completed = 0
        self._index_in_epoch = 0

    def get_data(self, data_dir, num_data_sec=180, fake=False):
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
        else:
            if num_data_sec == -1:
                data.append(np.random.randn(720, 18715))
            else:
                data.append(np.random.randn(720, 250000))


        # use very small dataset for overfitting test

        # create the data and normalize to [0,1]
        all_data = np.vstack(data)
        np.random.shuffle(all_data)

        self._test = all_data[:500]
        self._images = all_data[500:]

        #self._images = self._images[:64]

        #mean_data = np.mean(self._images)
        #std_data = np.std(self._images)

        max_val = np.max(self._images)
        min_val = np.min(self._images)
        self._images = np.divide((self._images - min_val), (max_val-min_val)) *2 - 1 
        self._test = np.divide((self._test - min_val), (max_val-min_val)) * 2 - 1

        #self._images = self._images / np.max(self._images)
        #self._images = (self._images - mean_data) / std_data
        #self._test = (self._test - mean_data) / std_data

        self._labels = None
        self._num_examples = self._images.shape[0]
        self._epochs_completed = 0
        self._index_in_epoch = 0

        # since the data is in stft, it's not clear how to normalize for NOW
        # normalize to [0,1]
        #self._images  = self._images / np.max(self._images)

    @property
    def epochs_completed(self):
        return self._epochs_completed

    @property
    def images(self):
        return self._images

    @property
    def test(self):
        return self._test

    @property
    def labels(self):
        return self._labels

    def read_hdf5(self, fName):
        f = h5py.File(fName, 'r')
        data = np.array(f['data'])
        return data


    def iterate_minibatches(self, batchsize, shuffle=True):
        if shuffle:
            indices = np.arange(self._images.shape[0])
            np.random.shuffle(indices)
        for start_idx in range(0, self._images.shape[0] - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield self._images[excerpt]


    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._imagesc = self._images[perm0]
            #self._labels = self._labels[perm0]

        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = None

            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self._images[perm]
                #self._labels = self._labels[perm]

            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            #labels_new_part = self._labels[start:end]
            labels_new_part = None
            return np.concatenate((images_rest_part, images_new_part), axis=0), None 
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], None

