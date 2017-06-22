#!/bin/bash

#python src/auto-encoder.py 
#    --data_dir='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/volumes_freqSum' 
#    --result_dir='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/results' 
#    --model='freqSumBig' 
#    --data_type='freqSum'

#python src/load-predict.py --data_dir='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/volumes_freqSum' --base_dir='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/results' --model='freqSumSmall' --data_type='freqSum' 

python src/load-predict.py --data_base_dir='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/processed_data' --trained_model_base_dir='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/trained_models' --model='freqSumBig' --trained_model_name='2017-06-21-230808' --data_type='freqSum'

#python src/load-predict.py --data_base_dir='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/processed_data' --trained_model_base_dir='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/trained_models' --model='freqSumBig' --trained_model_name='2017-06-20-095105' --data_type='freqSum' --meta_file='freqSumBig_epoch_500_2017-06-20-142652.ckpt-78500.meta'
