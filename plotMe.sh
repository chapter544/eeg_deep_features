#!/bin/bash

# freqSum_NoTiedWeight_Medium
python src/load-plot.py --model='freqSum_NoTiedWeight_Medium' --model_name='2017-07-16-023015' --data_base_dir='/home/chuong/EEG-Project/processed_data' --output_base_dir='/home/chuong/EEG-Project/output_features'

# freqSum_NoTiedWeight_Small
#python src/load-plot.py --model='freqSum_NoTiedWeight_Small' --model_name='2017-07-10-201227' --data_base_dir='/home/chuong/EEG-Project/processed_data' --output_base_dir='/home/chuong/EEG-Project/output_features'

# freqSum_NoTiedWeight_Big
#python src/load-plot.py --model='freqSum_NoTiedWeight_Big' --model_name='2017-06-24-030518' --data_base_dir='/home/chuong/EEG-Project/processed_data' --output_base_dir='/home/chuong/EEG-Project/output_features'
# freqSum_TiedWeight_Big model (8192 -> 128)
#python src/load-plot.py --model='freqSum_TiedWeight_Big' --model_name='2017-06-24-013834' --data_base_dir='/home/chuong/EEG-Project/processed_data' --output_base_dir='/home/chuong/EEG-Project/output_features'

# freqSum_TiedWeight model (4096 -> 128)
#python src/load-plot.py --model='freqSum_TiedWeight_Big' --model_name='2017-06-23-072749' --data_base_dir='/home/chuong/EEG-Project/processed_data' --output_base_dir='/home/chuong/EEG-Project/output_features'
