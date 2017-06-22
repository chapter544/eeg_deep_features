import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob, os


def read_and_parse_subject(data_dir):
    os.chdir(data_dir)
    for file in glob.glob("*.mat"):
        print file


fName = 'invcomp/invcomp100Hz_sub30075_15remout3.mat'

f = h5py.File(fName)
# f.keys() list the dictionary that the mat file contains

real_part = np.array(f['invres2dreal'])
imag_part = np.array(f['invres2dimag'])


# reshape to the 283x100 slices
A = np.reshape(real_part, (283, 100, -1))
B = np.reshape(imag_part, (283, 100, -1))

# test to see what it is
idx = 10
C = np.sqrt(A[:,:,idx]**2 + B[:,:,idx]**2)

# sampling on the cortex dimension
cortex_sampling = 10
AA = A[:,:,1::cortex_sampling]
BB = B[:,:,1::cortex_sampling]

# pick out a slice and display
idx = 3
a = AA[idx,:,:]
b = BB[idx,:,:]
c = np.sqrt(a**2 + b**2)
plt.imshow(c.transpose())
plt.show()
plt.savefig('blue_' + str(idx) + '.png')


