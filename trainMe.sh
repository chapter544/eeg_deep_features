#!/bin/bash

#python src/auto-encoder.py 
#    --data_dir='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/volumes_freqSum' 
#    --result_dir='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/results' 
#    --model='freqSumBig' 
#    --data_type='freqSum'

# freq 4-20 + tied weight
python src/auto-encoder.py --data_base_dir='/data1/volumes_freq_4_30' --trained_models_base_dir='/home/chuong/EEG-Project/trained_models' --model='freq_4_30_TiedWeight_Small' --data_type='freqSum' --batch_size=64 --num_epochs=400 --learning_rate=1e-3 --decay_rate=0.8 --decay_step=10000 --num_epochs_save=200 --gamma=1e-6 --data_normalization='normalize' --feature_activation='relu'


# big freqSum + NO tied weight
#python src/auto-encoder.py --data_base_dir='/home/chuong/volumes_freq_4_30' --trained_models_base_dir='/home/chuong/EEG-Project/trained_models' --model='freq_4_30_NoTiedWeight_Small' --data_type='freqSum' --batch_size=64 --num_epochs=2000 --learning_rate=1e-4 --decay_rate=0.8 --decay_step=10000 --num_epochs_save=999 --gamma=1e-6 --data_normalization='normalize' --feature_activation='relu'

# Tiny freqSum + no tied weight
#python src/auto-encoder.py --data_base_dir='/home/chuong/EEG-Project/processed_data' --trained_models_base_dir='/home/chuong/EEG-Project/trained_models' --model='freqSum_NoTiedWeight_Tiny' --data_type='freqSum' --batch_size=64 --num_epochs=2000 --learning_rate=1e-4 --decay_rate=0.8 --decay_step=10000 --num_epochs_save=999 --gamma=1e-6 --data_normalization='normalize'

# big freqSum + tied weight
#python src/auto-encoder.py --data_base_dir='/home/chuong/EEG-Project/processed_data' --trained_models_base_dir='/home/chuong/EEG-Project/trained_models' --model='freqSum_TiedWeight_Big' --data_type='freqSum' --batch_size=64 --num_epochs=2000 --learning_rate=1e-6 --decay_rate=0.8 --decay_step=10000 --num_epochs_save=999 --gamma=1e-6 --data_normalization='normalize'

# big freqSum + NO tied weight
#python src/auto-encoder.py --data_base_dir='/home/chuong/EEG-Project/processed_data' --trained_models_base_dir='/home/chuong/EEG-Project/trained_models' --model='freqSum_NoTiedWeight_Big' --data_type='freqSum' --batch_size=64 --num_epochs=4000 --learning_rate=1e-5 --decay_rate=0.8 --decay_step=10000 --num_epochs_save=999 --gamma=1e-6 --data_normalization='normalize' --feature_activation='linear'



# small freqSum + no tied weight + medium
#python src/auto-encoder.py --data_base_dir='/home/chuong/EEG-Project/processed_data' --trained_models_base_dir='/home/chuong/EEG-Project/trained_models' --model='freqSum_NoTiedWeight_Medium' --data_type='freqSum' --batch_size=64 --num_epochs=4000 --learning_rate=1e-4 --decay_rate=0.8 --decay_step=10000 --num_epochs_save=2000 --gamma=1e-7 --data_normalization='normalize' --feature_activation='linear'

# no bias training
#python src/auto-encoder.py --data_dir='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/processed_data/volumes_freqSum' --trained_output='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/results' --model='freqSumBig' --data_type='freqSum' --batch_size=64 --num_epochs=1000 --learning_rate=1e-5 --num_epochs_save=200 --gamma=1e-7

#python src/auto-encoder.py --data_dir='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/processed_data/volumes_freqSum' --trained_output='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/results' --model='freqSumBig' --data_type='freqSum' --batch_size=64 --num_epochs=1000 --learning_rate=1e-4 --num_epochs_save=200 --gamma=1e-7


# L1 with weight
#python src/auto-encoder.py --data_dir='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/processed_data/volumes_freqSum' --trained_output='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/results' --model='freqSumBig' --data_type='freqSum' --batch_size=64 --num_epochs=500 --learning_rate=1e-4 --num_epochs_save=100 --gamma=1e-7

#python src/auto-encoder.py --data_dir='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/processed_data/volumes_freqSum' --trained_output='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/results' --model='freqSumBig' --data_type='freqSum' --batch_size=32 --num_epochs=300 --learning_rate=1e-7 --num_epochs_save=50

#python src/auto-encoder.py --data_dir='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/processed_data/volumes_freqSum' --trained_output='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/results' --model='freqSumSmall' --data_type='freqSum' --gamma=1e-10 --batch_size=5 --num_epochs=20 --learning_rate=0.0001
