#!/bin/bash

#python src/auto-encoder.py 
#    --data_dir='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/volumes_freqSum' 
#    --result_dir='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/results' 
#    --model='freqSumBig' 
#    --data_type='freqSum'

# big freqSum + NO tied weight
python src/auto-encoder.py --data_base_dir='/data1/volumes_freq_4_30' --trained_models_base_dir='/home/chuong/EEG-Project/trained_models' --model='freq_4_30_NoTiedWeight_Small' --data_type='freqSum' --batch_size=256 --num_epochs=200 --learning_rate=1e-4 --num_epochs_save=100 --gamma=1e-6 --data_normalization='normalize' --feature_activation='relu'

# big freqSum + NO tied weight
#python src/auto-encoder.py --data_base_dir='/data1/volumes_freq_4_30' --trained_models_base_dir='/home/chuong/EEG-Project/trained_models' --model='freq_4_30_TiedWeight_Small' --data_type='freqSum' --batch_size=256 --num_epochs=200 --learning_rate=1e-4 --num_epochs_save=100 --gamma=1e-6 --data_normalization='normalize' --feature_activation='relu'
