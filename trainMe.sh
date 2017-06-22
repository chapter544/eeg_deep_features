#!/bin/bash

#python src/auto-encoder.py 
#    --data_dir='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/volumes_freqSum' 
#    --result_dir='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/results' 
#    --model='freqSumBig' 
#    --data_type='freqSum'

# no bias training
python src/auto-encoder.py --data_dir='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/processed_data/volumes_freqSum' --trained_output='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/results' --model='freqSumBig' --data_type='freqSum' --batch_size=64 --num_epochs=1000 --learning_rate=1e-5 --num_epochs_save=200 --gamma=1e-7

#python src/auto-encoder.py --data_dir='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/processed_data/volumes_freqSum' --trained_output='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/results' --model='freqSumBig' --data_type='freqSum' --batch_size=64 --num_epochs=1000 --learning_rate=1e-4 --num_epochs_save=200 --gamma=1e-7


# L1 with weight
#python src/auto-encoder.py --data_dir='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/processed_data/volumes_freqSum' --trained_output='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/results' --model='freqSumBig' --data_type='freqSum' --batch_size=64 --num_epochs=500 --learning_rate=1e-4 --num_epochs_save=100 --gamma=1e-7

#python src/auto-encoder.py --data_dir='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/processed_data/volumes_freqSum' --trained_output='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/results' --model='freqSumBig' --data_type='freqSum' --batch_size=32 --num_epochs=300 --learning_rate=1e-7 --num_epochs_save=50

#python src/auto-encoder.py --data_dir='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/processed_data/volumes_freqSum' --trained_output='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/results' --model='freqSumSmall' --data_type='freqSum' --gamma=1e-10 --batch_size=5 --num_epochs=20 --learning_rate=0.0001
