import pickle
import argparse
import os
from utils import get_input_data_path, get_data_path_with_timestamp


from datetime import datetime
import argparse

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt 

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, 
	default='big', 
	help='Architecture (big, small, freqSumSmall, freqSumBig)')
parser.add_argument('--model_name', type=str, 
	default='', 
	help='Trained model name')
parser.add_argument('--data_base_dir', type=str, 
	default='', 
	help='Data base directory')
parser.add_argument('--output_base_dir', type=str, 
	default='/home/chuong/EEG-Project/output_features', 
	help='Output base directory')

FLAGS, unparsed = parser.parse_known_args()

if FLAGS.model_name == '':
	raise Exception("A trained model name is required.")

model = FLAGS.model
sub_volumes_dir = get_input_data_path(FLAGS.model, FLAGS.data_base_dir)

# feature directory
feature_fullpath = FLAGS.output_base_dir + '/' + FLAGS.model + '-' + FLAGS.model_name


os.chdir(feature_fullpath)

#no_bias_dir = 'feature_no_bias'
#if not os.path.exists(no_bias_dir):
#	os.makedirs(no_bias_dir)

# no-bias load
#no_bias_pkl = feature_fullpath + '/' + 'volumes_time_feature_no_bias.pkl' 
#with open(no_bias_pkl, 'rb') as out_file: 
#    data_no_bias = pickle.load(out_file) 

#print("Plotting and saving no-bias images")
## no-bias plot
#for key, value in data_no_bias.iteritems(): 
#    plt.imshow(value, aspect='auto') 
#    plt.colorbar() 
#    plt.tight_layout() 
#    plt.savefig(no_bias_dir + '/' + key + '.png') 
#    plt.close()

with_bias_dir = 'feature_with_bias'
if not os.path.exists(with_bias_dir):
	os.makedirs(with_bias_dir)

with_bias_pkl = feature_fullpath + '/' + 'volumes_time_feature_with_bias.pkl' 
with open(with_bias_pkl, 'rb') as out_file: 
    data_with_bias = pickle.load(out_file) 

print("Plotting and saving with bias images")
for key, value in data_with_bias.iteritems(): 
    plt.imshow(value, aspect='auto') 
    plt.colorbar() 
    plt.tight_layout() 
    plt.savefig(with_bias_dir + '/' + key + '.png') 
    plt.close() 
