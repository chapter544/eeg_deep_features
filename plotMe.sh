#!/bin/bash

# freqSum_TiedWeight_Big model (8192 -> 128)
python src/load-plot.py --model='freqSum_TiedWeight_Big' --model_name='2017-06-23-072749' --data_base_dir='/home/chuong/EEG-Project/processed_data' --output_base_dir='/home/chuong/EEG-Project/output_features'

# freqSum_TiedWeight model (4096 -> 128)
#python src/load-plot.py --model='freqSum_TiedWeight_Big' --model_name='2017-06-23-072749' --data_base_dir='/home/chuong/EEG-Project/processed_data' --output_base_dir='/home/chuong/EEG-Project/output_features'
