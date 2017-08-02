import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob, os
import h5py


#fName = 'invcomp/invcomp100Hz_sub30075_15remout3.mat'
def subsample_mat_volume_freqSum(input_file, out_dir=None):
    f = h5py.File(input_file)
    # f.keys() list the dictionary that the mat file contains

    real_part = np.array(f['invres2dreal'], dtype=np.float32)
    imag_part = np.array(f['invres2dimag'], dtype=np.float32)

    # reshape to the 283x100 slices 
    A = np.reshape(real_part, (-1, 100, 18715))
    B = np.reshape(imag_part, (-1, 100, 18715))

    C = np.sqrt(A**2 + B**2)
    #D = np.sum(C, axis=1)

    outFile = input_file + '_freqSum.h5'
    #if out_dir is not None:
    #    outFile = out_dir + '/' + outFile

    h5file = h5py.File(outFile, mode='w')
    h5file.create_dataset("data", data=D) 
    h5file.close()


def subsample_5_freq_mat_volume(input_file, cortex_sample_rate, out_dir=None):
    f = h5py.File(input_file)
    # f.keys() list the dictionary that the mat file contains

    real_part = np.array(f['invres2dreal'], dtype=np.float32)
    imag_part = np.array(f['invres2dimag'], dtype=np.float32)

    # reshape to the 283x100 slices 
    A = np.reshape(real_part, (-1, 100, 18715))
    B = np.reshape(imag_part, (-1, 100, 18715))

    AA = A[:,1:50,::cortex_sample_rate]
    BB = B[:,1:50,::cortex_sample_rate]
    C = np.sqrt(AA**2 + BB**2)

    d1 = np.sum(C[:,0:2,:], axis=1) # delta
    d2 = np.sum(C[:,3:6,:], axis=1) # theta
    d3 = np.sum(C[:,7:12,:], axis=1) # alpha
    d4 = np.sum(C[:,13:29,:], axis=1) # beta
    d5 = np.sum(C[:,30:49,:], axis=1) # gamma

    D = np.concatenate((d1, d2, d3, d4, d5), axis=1)
    E = D.reshape(D.shape[0],-1)

    outFile = input_file + '_freq_5_cortexsample' + \
              str(cortex_sample_rate) + '.h5'
    if out_dir is not None:
        outFile = out_dir + '/' + os.path.basename(outFile)

    h5file = h5py.File(outFile, mode='w')
    h5file.create_dataset("data", data=E) 
    h5file.close()





def subsample_3_40_freq_mat_volume(input_file, 
        start_freq=4, end_freq=30, 
        cortex_sample_rate=1, out_dir=None):
    f = h5py.File(input_file)
    # f.keys() list the dictionary that the mat file contains

    real_part = np.array(f['invres2dreal'], dtype=np.float32)
    imag_part = np.array(f['invres2dimag'], dtype=np.float32)

    # reshape to the 283x100 slices 
    A = np.reshape(real_part, (-1, 100, 18715))
    B = np.reshape(imag_part, (-1, 100, 18715))

    AA = A[:,start_freq-1:end_freq,::cortex_sample_rate]
    BB = B[:,start_freq-1:end_freq,::cortex_sample_rate]
    C = np.sqrt(AA**2 + BB**2)
    C = C.reshape(C.shape[0],-1)

    outFile = input_file + '_freq' + str(start_freq) + '_' + \
              str(end_freq) + '_cortexsample' + \
              str(cortex_sample_rate) + '.h5'
    if out_dir is not None:
        outFile = out_dir + '/' + os.path.basename(outFile)

    h5file = h5py.File(outFile, mode='w')
    h5file.create_dataset("data", data=C) 
    h5file.close()



def subsample_mat_volume(input_file, 
        start_time_sec=30, end_time_sec=150, 
        freq_trunc=40, cortex_sample_rate=4, out_dir=None):
    f = h5py.File(input_file)
    # f.keys() list the dictionary that the mat file contains

    real_part = np.array(f['invres2dreal'], dtype=np.float32)
    imag_part = np.array(f['invres2dimag'], dtype=np.float32)

    # reshape to the 283x100 slices 
    A = np.reshape(real_part, (-1, 100, 18715))
    B = np.reshape(imag_part, (-1, 100, 18715))

    # end_time_sec is the last entry
    if end_time_sec == None:
        end_time_sec = A.shape[0]

    AA = A[start_time_sec:end_time_sec,:freq_trunc,::cortex_sample_rate]
    BB = B[start_time_sec:end_time_sec,:freq_trunc,::cortex_sample_rate]
    C = np.sqrt(AA**2 + BB**2)

    outFile = input_file + '_time' + str(start_time_sec) + '_' + \
              str(end_time_sec) + '_freq' + str(freq_trunc) + '_sample' + \
              str(cortex_sample_rate) + '.h5'
    if out_dir is not None:
        outFile = out_dir + '/' + outFile

    h5file = h5py.File(outFile, mode='w')
    h5file.create_dataset("data", data=C) 
    h5file.close()


    #idxs = [i for i in range(0, 10) if i % 2 == 0]
    #for idx in idxs:
    #    # sampling on the cortex dimension
    #    AA = A[:,:freq_trunc,idx::cortex_sample_rate]
    #    BB = B[:,:freq_trunc,idx::cortex_sample_rate]
    #    C = np.sqrt(AA**2 + BB**2)
    #
    #    outFile = input_file + '_' + str(idx) + '.h5'
    #    h5file = h5py.File(outFile, mode='w')
    #    h5file.create_dataset("data", data=C) 
    #    h5file.close()

def check_subject_cortex_size(data_dir):
    os.chdir(data_dir)
    for fName in glob.glob("*.mat"):
        f = h5py.File(data_dir + '/' + fName)
        # f.keys() list the dictionary that the mat file contains
        print f['invres2dreal']


def check_subject_cortex_images(data_dir):
    os.chdir(data_dir)
    for fName in glob.glob("*.mat"):
        data = h




def read_hdf5(fName):
    f = h5py.File(fName, 'r')
    data = np.array(f['data'])
    return data


def generate_subsample_volumes(data_dir, out_dir):
    os.chdir(data_dir)
    for fName in glob.glob("*.mat"):
        freq_trunc=40
        cortex_sample_rate=3
        start_time_sec = 30
        end_time_sec = 210
        subsample_mat_volume(data_dir + '/' + fName, start_time_sec, end_time_sec, freq_trunc, cortex_sample_rate)


def generate_subsample_volumes_except_time(data_dir, out_dir):
    os.chdir(data_dir)
    for fName in glob.glob("*.mat"):
        freq_trunc=40
        cortex_sample_rate=3
        start_time_sec = 0
        end_time_sec = None
        subsample_mat_volume(data_dir + '/' + fName, start_time_sec, end_time_sec, freq_trunc, cortex_sample_rate)


def generate_subsample_volumes_freqSum(data_dir, out_dir):
    os.chdir(data_dir)
    for fName in glob.glob("*.mat"):
        print("Doing %s ..." % fName)
        subsample_mat_volume_freqSum(data_dir + '/' + fName, out_dir)


def generate_volumes_freq_4_30(data_dir, out_dir):
    os.chdir(data_dir)

    for fName in glob.glob("*.mat"):
        start_freq=4
        end_freq=30
        cortex_sample_rate=3
        subsample_3_40_freq_mat_volume(
                data_dir + '/' + fName, 
                start_freq, end_freq, 
                cortex_sample_rate, 
                out_dir)

def generate_volumes_freq_5(data_dir, out_dir):
    os.chdir(data_dir)

    cortex_sample_rate = 1
    for fName in glob.glob("*.mat"):
        subsample_5_freq_mat_volume(data_dir + '/' + fName, 
                                    cortex_sample_rate, out_dir)



def generate_subsample_a_volume(data_dir, start_time_sec, end_time_sec, freq_trunc, cortex_sample_rate):
    fName = 'invcomp100Hz_sub30152_9remout3.mat'
    subsample_mat_volume(data_dir + '/' + fName, start_time_sec, end_time_sec, freq_trunc, cortex_sample_rate)



def main():
    #data_dir = '/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/invcomp'
    data_dir = '/home/chuong/EEG-Project/invcomp'

    # Generate subvolumes with sampling on both time, frequency, and 
    # cortex (space) axes.
    #out_dir = '/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/sub_volumes'
    #generate_subsample_volumes(data_dir, out_dir)

    #out_dir = '/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/volumes_freqSum'
    #out_dir = '/home/chuong/volumes_freq_4_30'
    #generate_volumes_freq_4_30(data_dir, out_dir)


    out_dir = '/home/chuong/volumes_freq_5'
    generate_volumes_freq_5(data_dir, out_dir)


    #generate_subsample_volumes_freqSum(data_dir, out_dir)

    # Generate subvolumes with sampling on both time, frequency, and 
    # cortex (space) axes.
    #out_dir = '/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/sub_volumes_except_time'
    #generate_subsample_volumes_except_time(data_dir, out_dir)

    #generate_subsample_a_volume(data_dir, 30, 210, 40, 3)

    #check_subject_cortex_size(data_dir)

    # check images
    #check_subject_cortex_images(data_dir)


if __name__ == '__main__':
    main()

