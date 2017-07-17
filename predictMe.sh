#!/bin/bash

#python src/auto-encoder.py 
#    --data_dir='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/volumes_freqSum' 
#    --result_dir='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/results' 
#    --model='freqSumBig' 
#    --data_type='freqSum'

#python src/load-predict.py --data_dir='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/volumes_freqSum' --base_dir='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/results' --model='freqSumSmall' --data_type='freqSum' 

# Small freqSum + no tied weight + Medium size
python src/load-predict.py --data_base_dir='/home/chuong/EEG-Project/processed_data' --trained_model_base_dir='/home/chuong/EEG-Project/trained_models' --model='freqSum_NoTiedWeight_Medium' --trained_model_name='2017-07-16-023015' --data_type='freqSum' --data_normalization='normalize'

# Small freqSum + no tied weight + Medium size
#python src/load-predict.py --data_base_dir='/home/chuong/EEG-Project/processed_data' --trained_model_base_dir='/home/chuong/EEG-Project/trained_models' --model='freqSum_NoTiedWeight_Medium' --trained_model_name='2017-07-11-061415' --data_type='freqSum'

# Small freqSum + no tied weight
#python src/load-predict.py --data_base_dir='/home/chuong/EEG-Project/processed_data' --trained_model_base_dir='/home/chuong/EEG-Project/trained_models' --model='freqSum_NoTiedWeight_Small' --trained_model_name='2017-07-10-201227' --data_type='freqSum'

# big freqSum + no tied weight
#python src/load-predict.py --data_base_dir='/home/chuong/EEG-Project/processed_data' --trained_model_base_dir='/home/chuong/EEG-Project/trained_models' --model='freqSum_NoTiedWeight_Big' --trained_model_name='2017-06-24-030518' --data_type='freqSum'
#python src/load-predict.py --data_base_dir='/home/chuong/EEG-Project/processed_data' --trained_model_base_dir='/home/chuong/EEG-Project/trained_models' --model='freqSum_TiedWeight_Big' --trained_model_name='2017-06-24-013834' --data_type='freqSum'

#python src/load-predict.py --data_base_dir='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/processed_data' --trained_model_base_dir='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/trained_models' --model='freqSumBig' --trained_model_name='2017-06-20-095105' --data_type='freqSum' --meta_file='freqSumBig_epoch_500_2017-06-20-142652.ckpt-78500.meta'
