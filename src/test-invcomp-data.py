import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob, os

data_dir = '/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/invcomp'
fName = 'invcomp100Hz_sub30152_9remout3.mat'
input_file = data_dir + '/' + fName

f = h5py.File(input_file)

real_part = np.array(f['invres2dreal'], dtype=np.float32)
imag_part = np.array(f['invres2dimag'], dtype=np.float32)

fft_mag = np.sqrt(real_part ** 2 + imag_part ** 2)

print("Mean: {}, Max: {}, Min: {}, STD: {}".format(np.mean(fft_mag),
    np.max(fft_mag), np.min(fft_mag), np.std(fft_mag)))

#A = real_part[0:100, 100]
#B = imag_part[0:100, 100]
#C = A + 1j * B
#c = np.fft.ifft(C, 512)
#plt.plot(range(0, len(c)), np.real(c))
#plt.show()


#A1 = real_part[101:200, 100]
#B1 = imag_part[101:200, 100]
#C1 = A1 + 1j * B1
#c1 = np.fft.ifft(C1, 512)
#plt.plot(range(0, len(c1)), np.real(c1))
#plt.show()


c2 = np.concatenate( (np.real(c), np.real(c1)) )
plt.plot(range(0, len(c2)), np.real(c2))
plt.show()




#c = np.sqrt(a**2 + b**2)
#plt.imshow(c)
#plt.show()

#cc = c.reshape((-1, 100, 100))
#
#c1 = cc[0,:,:]
#c0 = c[0:100,:]
