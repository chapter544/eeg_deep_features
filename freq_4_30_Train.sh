#!/bin/bash

#python src/auto-encoder.py 
#    --data_dir='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/volumes_freqSum' 
#    --result_dir='/data1/CHUONG_DATA/ChuongWork/Data4DeepLearning/results' 
#    --model='freqSumBig' 
#    --data_type='freqSum'


# freq 4-30 + NO tied weight + Behind + BN 
python src/auto-encoder.py \
    --data_base_dir='/data1/volumes_freq_4_30' \
    --trained_models_base_dir='/home/chuong/EEG-Project/trained_models' \
    --model='freq_4_30_NoTiedWeight' \
    --data_type='freqSum'\
    #--network_params="400,400,400,200,200,64" \
    --network_params="400,200,200,64" \
    --use_BN=1 \
    --use_BN_Contrib=0 \
    --use_BN_Front=0 \
    --use_dropout=0 \
    --dropout_keep=0.7 \
    --use_L1_Reg=0 \
    --gamma=1e-7 \
    --data_normalization='normalize' \
    --feature_activation='elu' \
    --batch_size=1000 \
    --learning_rate=1e-3 \
    --lr_intervals="3000,6000" \
    --num_epochs=1000 \
    --num_epochs_save=1000


