
# Auto-encoder for 2D cortex slices
# Chuong Nguyen
# 07/11/2017
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from pkg_resources import parse_version
import tensorflow as tf
import numpy as np
import sys
from tf_utils import build_fc_layer


def build_network_SumAll_NoTiedWeight(
        X, 
        network_dims,
        bn_is_training,
        use_dropout=False,
        keep_prob=0.5, 
        use_BN=False,
        use_BN_Contrib=False,
        use_BN_Front=False,
        use_L1_Reg=False,
        gamma=1e-7, 
        activation='elu'):

    # setting network input and output dimensions
    X_dim, fc1_dim, fc2_dim, fc3_dim, fc3a_dim, fc4_dim, fc_feat_dim = network_dims
    X_dim, fc8_dim, fc7_dim, fc6_dim, fc5a_dim, fc5_dim, fc_feat_dim = network_dims

    # FC1
    fc1 = build_fc_layer(X, X_dim, fc1_dim, 'FC1',
                            bn_is_training,
                            use_dropout=use_dropout, 
                            #use_dropout=False, 
                            keep_prob=keep_prob,
                            use_BN=use_BN,
                            use_BN_Contrib=use_BN_Contrib,
                            use_BN_Front=use_BN_Front, activation='elu')
    # FC2
    fc2 = build_fc_layer(fc1, fc1_dim, fc2_dim, 'FC2',
                            bn_is_training,
                            use_dropout=use_dropout, 
                            #use_dropout=False, 
                            keep_prob=keep_prob,
                            use_BN=use_BN,
                            use_BN_Contrib=use_BN_Contrib,
                            #bn_is_training=bn_is_training, 
                            use_BN_Front=use_BN_Front, activation='elu')

    # FC3
    fc3 = build_fc_layer(fc2, fc2_dim, fc3_dim, 'FC3',
                            bn_is_training,
                            #use_dropout=use_dropout, 
                            use_dropout=False, 
                            keep_prob=keep_prob,
                            use_BN=use_BN,
                            use_BN_Contrib=use_BN_Contrib,
							use_BN_Front=use_BN_Front, activation='elu')

    # FC3
    fc3a = build_fc_layer(fc3, fc3_dim, fc3a_dim, 'FC3a',
                            bn_is_training,
                            #use_dropout=use_dropout, 
                            use_dropout=False, 
                            keep_prob=keep_prob,
                            use_BN=use_BN,
                            use_BN_Contrib=use_BN_Contrib,
                            use_BN_Front=use_BN_Front, activation='elu')

    # FC4
    fc4 = build_fc_layer(fc3a, fc3a_dim, fc4_dim, 'FC4',
                            bn_is_training,
                            #use_dropout=use_dropout, 
                            use_dropout=False, 
                            keep_prob=keep_prob,
                            use_BN=use_BN,
                            use_BN_Contrib=use_BN_Contrib,
                            use_BN_Front=use_BN_Front, activation='elu')
    # FCFeat
    fc_feat = build_fc_layer(fc4, fc4_dim, fc_feat_dim, 'FCFeat',
                            bn_is_training,
                            #use_dropout=use_dropout, 
                            use_dropout=False, 
                            keep_prob=keep_prob,
                            use_BN=use_BN,
                            #use_BN=False, # NO BN on the feature output
                            use_BN_Contrib=use_BN_Contrib,
                            use_BN_Front=use_BN_Front, activation='elu')
    # FC5
    fc5 = build_fc_layer(fc_feat, fc_feat_dim, fc5_dim, 'FC5',
                            bn_is_training,
                            use_dropout=False, 
                            #use_dropout=use_dropout, 
                            keep_prob=keep_prob,
                            use_BN=use_BN,
                            use_BN_Contrib=use_BN_Contrib,
                            use_BN_Front=use_BN_Front, activation='elu')
    # FC5
    fc5a = build_fc_layer(fc5, fc5_dim, fc5a_dim, 'FC5a',
                            bn_is_training,
                            #use_dropout=use_dropout, 
                            use_dropout=False, 
                            keep_prob=keep_prob,
                            use_BN=use_BN,
                            use_BN_Contrib=use_BN_Contrib,
                            use_BN_Front=use_BN_Front, activation='elu')

    # FC6
    fc6 = build_fc_layer(fc5a, fc5a_dim, fc6_dim, 'FC6',
                            bn_is_training,
                            #use_dropout=use_dropout, 
                            use_dropout=False, 
                            keep_prob=keep_prob,
                            use_BN=use_BN,
                            use_BN_Contrib=use_BN_Contrib,
                            use_BN_Front=use_BN_Front, activation='elu')
 
    # FC7
    fc7 = build_fc_layer(fc6, fc6_dim, fc7_dim, 'FC7',
                            bn_is_training,
                            use_dropout=use_dropout, 
                            keep_prob=keep_prob,
                            use_BN=use_BN,
                            use_BN_Contrib=use_BN_Contrib,
                            use_BN_Front=use_BN_Front, activation='elu')
    # FC8
    fc8 = build_fc_layer(fc7, fc7_dim, fc8_dim, 'FC8',
                            bn_is_training,
                            use_dropout=use_dropout, 
                            keep_prob=keep_prob,
                            use_BN=use_BN,
                            use_BN_Contrib=use_BN_Contrib,
                            use_BN_Front=use_BN_Front, activation='elu')
 
    # FC9
    fc9 = build_fc_layer(fc8, fc8_dim, X_dim, 'FC9',
                            bn_is_training,
                            use_dropout=False, 
                            keep_prob=keep_prob,
                            use_BN=False, # NO BN on the last layer
                            use_BN_Contrib=use_BN_Contrib,
                            #bn_is_training=bn_is_training, 
                            use_BN_Front=use_BN_Front, 
                            activation='none') # LINEAR ACTIVATION)
 
    # LOSS 
    with tf.name_scope("loss"):
        y = fc9 + 1e-10
        l2_loss = tf.sqrt(tf.reduce_mean(tf.square(y-X)))
        loss = l2_loss
        tf.summary.scalar("loss", loss)
        summary_op = tf.summary.merge_all()

    return loss, y, loss
