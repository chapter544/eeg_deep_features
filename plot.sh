#!/bin/bash

#python src/auto-encoder.py 
#    --data_dir='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/volumes_freqSum' 
#    --result_dir='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/results' 
#    --model='freqSumBig' 
#    --data_type='freqSum'

# freq_4_30_TiedWeight_Small
#python src/load-and-predict.py --data_base_dir='/home/chuong/EEG-Project/processed_data' --trained_model_base_dir='/home/chuong/EEG-Project/trained_models' --model='freq_4_30_TiedWeight_Small' --trained_model_name='2017-08-09-193907-relu' --data_type='freqSum' --data_normalization='normalize' --feature_activation='linear'

# freq_4_30_NoTiedWeight_Small
#python src/load-and-predict.py --data_base_dir='/home/chuong/EEG-Project/processed_data' --trained_model_base_dir='/home/chuong/EEG-Project/trained_models' --model='freq_4_30_NoTiedWeight_Small' --trained_model_name='2017-08-10-120651-relu' --data_type='freqSum' --data_normalization='normalize' --feature_activation='linear'
###############################################################################
# freqSum_NoTiedWeight_Small
#python src/load-and-predict.py --data_base_dir='/home/chuong/EEG-Project/processed_data' --trained_model_base_dir='/home/chuong/EEG-Project/trained_models' --model='freqSum_NoTiedWeight_Small' --trained_model_name='2017-08-10-160853-elu' --data_type='freqSum' --data_normalization='normalize' --feature_activation='linear'


#python src/load-predict.py --data_base_dir='/home/chuong/EEG-Project/processed_data' --trained_model_base_dir='/home/chuong/EEG-Project/trained_models' --model='freqSum_NoTiedWeight_Big' --trained_model_name='2017-07-18-052148' --data_type='freqSum' --data_normalization='normalize' --feature_activation='linear'

###############################################################################

# freq_5_TiedWeight_L1_Small
python src/load-and-predict.py --data_base_dir='/home/chuong/EEG-Project/processed_data' --trained_model_base_dir='/home/chuong/EEG-Project/trained_models' --model='freq_5_TiedWeight_L1_Small' --trained_model_name='2017-08-10-140210-elu' --data_type='freqSum' --data_normalization='normalize' --feature_activation='linear'

# freq_5_NoTiedWeight_Big
python src/load-and-predict.py --data_base_dir='/home/chuong/EEG-Project/processed_data' --trained_model_base_dir='/home/chuong/EEG-Project/trained_models' --model='freq_5_NoTiedWeight_Big' --trained_model_name='2017-08-06-200625-relu' --data_type='freqSum' --data_normalization='normalize' --feature_activation='linear'


# freq_5_NoTiedWeight_Small
python src/load-and-predict.py --data_base_dir='/home/chuong/EEG-Project/processed_data' --trained_model_base_dir='/home/chuong/EEG-Project/trained_models' --model='freq_5_NoTiedWeight_Small' --trained_model_name='2017-08-10-130927-elu' --data_type='freqSum' --data_normalization='normalize' --feature_activation='linear'

# freq_5_NoTiedWeight_L1_Small
python src/load-and-predict.py --data_base_dir='/home/chuong/EEG-Project/processed_data' --trained_model_base_dir='/home/chuong/EEG-Project/trained_models' --model='freq_5_NoTiedWeight_L1_Small' --trained_model_name='2017-08-08-032641-relu' --data_type='freqSum' --data_normalization='normalize' --feature_activation='linear'


# freq_5_TiedWeight_Big
python src/load-and-predict.py --data_base_dir='/home/chuong/EEG-Project/processed_data' --trained_model_base_dir='/home/chuong/EEG-Project/trained_models' --model='freq_5_TiedWeight_Big' --trained_model_name='2017-08-06-100646-relu' --data_type='freqSum' --data_normalization='normalize' --feature_activation='linear'
