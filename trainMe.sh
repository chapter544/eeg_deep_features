#!/bin/bash

#python src/auto-encoder.py 
#    --data_dir='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/volumes_freqSum' 
#    --result_dir='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/results' 
#    --model='freqSumBig' 
#    --data_type='freqSum'

<<<<<<< HEAD
# small freqSum + no tied weight + medium
python src/auto-encoder.py --data_dir='/home/chuong/EEG-Project/volumes_freqSum' --trained_models_base_dir='/home/chuong/EEG-Project/trained_models' --model='freqSum_NoTiedWeight_Medium' --data_type='freqSum' --batch_size=64 --num_epochs=1000 --learning_rate=1e-4 --decay_rate=0.8 --decay_step=10000 --num_epochs_save=300 --gamma=1e-7

# no bias training
#python src/auto-encoder.py --data_dir='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/processed_data/volumes_freqSum' --trained_output='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/results' --model='freqSumBig' --data_type='freqSum' --batch_size=64 --num_epochs=1000 --learning_rate=1e-5 --num_epochs_save=200 --gamma=1e-7
=======
# big freqSum + no tied weight
#python src/auto-encoder.py --data_dir='/home/chuong/EEG-Project/volumes_freqSum' --trained_models_base_dir='/home/chuong/EEG-Project/trained_models' --model='freqSum_NoTiedWeight_Big' --data_type='freqSum' --batch_size=64 --num_epochs=200 --learning_rate=1e-4 --decay_rate=0.8 --decay_step=5000 --num_epochs_save=50 --gamma=1e-5

# small freqSum + no tied weight + medium
python src/auto-encoder.py --data_dir='/home/chuong/EEG-Project/volumes_freqSum' --trained_models_base_dir='/home/chuong/EEG-Project/trained_models' --model='freqSum_NoTiedWeight_Medium' --data_type='freqSum' --batch_size=64 --num_epochs=1000 --learning_rate=1e-4 --decay_rate=0.8 --decay_step=10000 --num_epochs_save=300 --gamma=1e-7

# small freqSum + no tied weight
#python src/auto-encoder.py --data_dir='/home/chuong/EEG-Project/volumes_freqSum' --trained_models_base_dir='/home/chuong/EEG-Project/trained_models' --model='freqSum_NoTiedWeight_Small' --data_type='freqSum' --batch_size=64 --num_epochs=1000 --learning_rate=1e-4 --decay_rate=0.8 --decay_step=10000 --num_epochs_save=300 --gamma=1e-7
# tied weight
#python src/auto-encoder.py --data_dir='/home/chuong/EEG-Project/volumes_freqSum' --trained_models_base_dir='/home/chuong/EEG-Project/trained_models' --model='freqSum_TiedWeight_Big' --data_type='freqSum' --batch_size=64 --num_epochs=400 --learning_rate=1e-4 --decay_rate=0.8 --decay_step=5000 --num_epochs_save=100 --gamma=1e-6
>>>>>>> a3ed0a30cbe6b19704b4556dc6cf862fcf9e321f

#python src/auto-encoder.py --data_dir='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/processed_data/volumes_freqSum' --trained_output='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/results' --model='freqSumBig' --data_type='freqSum' --batch_size=64 --num_epochs=1000 --learning_rate=1e-4 --num_epochs_save=200 --gamma=1e-7


# L1 with weight
#python src/auto-encoder.py --data_dir='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/processed_data/volumes_freqSum' --trained_output='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/results' --model='freqSumBig' --data_type='freqSum' --batch_size=64 --num_epochs=500 --learning_rate=1e-4 --num_epochs_save=100 --gamma=1e-7

#python src/auto-encoder.py --data_dir='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/processed_data/volumes_freqSum' --trained_output='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/results' --model='freqSumBig' --data_type='freqSum' --batch_size=32 --num_epochs=300 --learning_rate=1e-7 --num_epochs_save=50

#python src/auto-encoder.py --data_dir='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/processed_data/volumes_freqSum' --trained_output='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/results' --model='freqSumSmall' --data_type='freqSum' --gamma=1e-10 --batch_size=5 --num_epochs=20 --learning_rate=0.0001
