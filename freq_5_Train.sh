#!/bin/bash

#python src/auto-encoder.py 
#    --data_dir='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/volumes_freqSum' 
#    --result_dir='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/results' 
#    --model='freqSumBig' 
#    --data_type='freqSum'

# freq 5 + tied weight + L1 + TINY
#python src/auto-encoder.py --data_base_dir='/home/chuong/volumes_freq_5' --trained_models_base_dir='/home/chuong/EEG-Project/trained_models' --model='freq_5_TiedWeight_L1_Tiny' --data_type='freqSum' --batch_size=128 --num_epochs=300 --learning_rate=1e-4 --num_epochs_save=150 --gamma=1e-8 --data_normalization='normalize' --feature_activation='elu'

# freq 5 + tied weight
#python src/auto-encoder.py --data_base_dir='/data1/volumes_freq_5' --trained_models_base_dir='/home/chuong/EEG-Project/trained_models' --model='freq_5_TiedWeight_Small' --data_type='freqSum' --batch_size=256 --num_epochs=200 --learning_rate=1e-4 --num_epochs_save=100 --gamma=1e-6 --data_normalization='normalize' --feature_activation='elu'

#python src/auto-encoder.py --data_base_dir='/data1/volumes_freq_5' --trained_models_base_dir='/home/chuong/EEG-Project/trained_models' --model='freq_5_TiedWeight_Small' --data_type='freqSum' --batch_size=256 --num_epochs=200 --learning_rate=1e-4 --num_epochs_save=100 --gamma=1e-6 --data_normalization='normalize' --feature_activation='elu'

# freq 5 + NO tied weight + L1 + TINY + BN
#python src/auto-encoder.py --data_base_dir='/home/chuong/volumes_freq_5' --trained_models_base_dir='/home/chuong/EEG-Project/trained_models' --model='freq_5_NoTiedWeight_L1_BN_Tiny' --data_type='freqSum' --batch_size=128 --num_epochs=200 --learning_rate=1e-4 --num_epochs_save=100 --gamma=1e-7 --data_normalization='normalize' --feature_activation='elu'


# freq 5 + NO tied weight + L1 + TINY  -- REVERSE BN
python src/auto-encoder.py --data_base_dir='/home/chuong/volumes_freq_5' --trained_models_base_dir='/home/chuong/EEG-Project/trained_models' --model='freq_5_NoTiedWeight_L1_BN_Tiny' --data_type='freqSum' --batch_size=1000 --num_epochs=400 --learning_rate=0.0003 --num_epochs_save=400 --gamma=1e-5 --data_normalization='normalize' --feature_activation='elu'


# freq 5 + NO tied weight + L1 + SMALL + BN
#python src/auto-encoder.py --data_base_dir='/home/chuong/volumes_freq_5' --trained_models_base_dir='/home/chuong/EEG-Project/trained_models' --model='freq_5_NoTiedWeight_L1_BN_Small' --data_type='freqSum' --batch_size=128 --num_epochs=300 --learning_rate=1e-4 --num_epochs_save=150 --gamma=1e-7 --data_normalization='normalize' --feature_activation='elu'

# freq 5 + NO tied weight + L1 + SMALL
#python src/auto-encoder.py --data_base_dir='/home/chuong/volumes_freq_5' --trained_models_base_dir='/home/chuong/EEG-Project/trained_models' --model='freq_5_NoTiedWeight_L1_Small' --data_type='freqSum' --batch_size=128 --num_epochs=200 --learning_rate=1e-4 --num_epochs_save=100 --gamma=1e-8 --data_normalization='normalize' --feature_activation='elu'

# freq 5 + tied weight + L1 + SMALL
#python src/auto-encoder.py --data_base_dir='/home/chuong/volumes_freq_5' --trained_models_base_dir='/home/chuong/EEG-Project/trained_models' --model='freq_5_TiedWeight_L1_Small' --data_type='freqSum' --batch_size=128 --num_epochs=200 --learning_rate=1e-4 --num_epochs_save=100 --gamma=1e-8 --data_normalization='normalize' --feature_activation='elu'

# freq 5 SMALL + NO tied weight
#python src/auto-encoder.py --data_base_dir='/data1/volumes_freq_5' --trained_models_base_dir='/home/chuong/EEG-Project/trained_models' --model='freq_5_NoTiedWeight_Small' --data_type='freqSum' --batch_size=128 --num_epochs=200 --learning_rate=1e-4 --num_epochs_save=100 --gamma=1e-6 --data_normalization='normalize' --feature_activation='elu'

# freq 5 SMALL + tied weight
#python src/auto-encoder.py --data_base_dir='/data1/volumes_freq_5' --trained_models_base_dir='/home/chuong/EEG-Project/trained_models' --model='freq_5_TiedWeight_Small' --data_type='freqSum' --batch_size=256 --num_epochs=200 learning_rate=1e-4 --num_epochs_save=100 --gamma=1e-6 --data_normalization='normalize' --feature_activation='elu'

# freq 5 + NO tied weight + BIG
#python src/auto-encoder.py --data_base_dir='/data1/volumes_freq_5' --trained_models_base_dir='/home/chuong/EEG-Project/trained_models' --model='freq_5_NoTiedWeight_Big' --data_type='freqSum' --batch_size=256 --num_epochs=200 --learning_rate=1e-4 -num_epochs_save=100 --gamma=1e-6 --data_normalization='normalize' --feature_activation='elu'

# freq 5 + tied weight + BIG
#python src/auto-encoder.py --data_base_dir='/data1/volumes_freq_5' --trained_models_base_dir='/home/chuong/EEG-Project/trained_models' --model='freq_5_TiedWeight_Big' --data_type='freqSum' --batch_size=256 --num_epochs=200 --learning_rate=1e-4 --num_epochs_save=100 --gamma=1e-6 --data_normalization='normalize' --feature_activation='elu'

