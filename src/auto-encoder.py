# Auto-encoder for 2D cortex slices
# Chuong Nguyen
# 07/11/2017
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import h5py
import glob
import os
import argparse
import sys
from datetime import datetime
from pkg_resources import parse_version
from eeg_input_data import eeg_data
from eeg_org_data import eeg_subject_data
from utils import get_input_data_path, get_data_path_with_timestamp
from utils import build_model
from models.fc_freqSum_TiedWeight import build_fc_freqSum_TiedWeight
from models.fc_freqSum_TiedWeight import build_fc_freqSum_TiedWeight_NoBias
from models.fc_freqSum_TiedWeight import build_fc_freqSum_TiedWeight_NoDropout
from models.fc_freqSum_TiedWeight import build_fc_freqSum
from models.fc_freqSum_TiedWeight import build_fc_freqSum_TiedWeight_Big
from models.fc_freqSum_TiedWeight import build_fc_freqSum_NoTiedWeight_Big
from models.fc_freqSum_TiedWeight import build_fc_freqSum_NoTiedWeight_Small
#from models.fc_freqSum_TiedWeight import build_fc_freqSum_NoTiedWeight_Medium
from models.fc_freqSum_TiedWeight import build_fc_freqSum_NoTiedWeight_Tiny

from models.fc_freq_Models import build_fc_freq_4_30_TiedWeight_Small
from models.fc_freq_Models import build_fc_freq_4_30_NoTiedWeight_Small
from models.fc_freq_Models import build_fc_freq_5_TiedWeight_Small


FLAGS = None


def main(_):
    ##################################################################
    # Read data 
    sub_volumes_dir = get_input_data_path(FLAGS.model, FLAGS.data_base_dir)
    eeg = eeg_data()

    if FLAGS.data_type == 'subsample': # subsample on 3D axes
        eeg.get_data(sub_volumes_dir, fake=FLAGS.test)
    else: # no subsampling
        eeg.get_data(sub_volumes_dir, num_data_sec=-1, 
                fake=FLAGS.test, normalization=FLAGS.data_normalization)

    X = eeg.train_data
    print('{} x {}'.format(X.shape[0], X.shape[1]))
    x_dim = X.shape[1]

    # reset everything
    tf.reset_default_graph()
    model_path = FLAGS.trained_model_base_dir

    # Input placeholder for input variables
    with tf.name_scope("input"):
        x = tf.placeholder(tf.float32, [None, x_dim])
        dropout_keep_prob = tf.placeholder(tf.float32)
        is_training = tf.placeholder(tf.bool, [])


    # BUILD MODEL
    # L1 regularization gamma
    gamma = FLAGS.gamma
    feature_activation = FLAGS.feature_activation
    loss, decoded, l1_loss = build_model(
		FLAGS.model,
		x,
		x_dim,
		dropout_keep_prob,
		gamma,
		feature_activation,
		is_training)


    # create model directory to write outputs
    model_path = get_data_path_with_timestamp(
            FLAGS.model, FLAGS.trained_model_base_dir)
    model_path += '-' + feature_activation
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # write model logs for tensorboard
    logs_path = '/tmp/eeg/logs/' +  \
                FLAGS.model +  \
                '-' + \
                datetime.now().strftime('%Y-%m-%d-%H%M%S')
    model_file_prefix = model_path + '/' + FLAGS.model + '_epoch_'



    global_step = tf.Variable(0, name="global_step", trainable=False)

    # OPTIMIZER
     # AdamOptimizer
    lr_rate = FLAGS.learning_rate
    train_step = tf.train.AdamOptimizer(
                    learning_rate=lr_rate).\
                    minimize(loss, global_step=global_step) 


    # MomentumOptmizer
    #decay_rate = FLAGS.decay_rate
    #decay_step = FLAGS.decay_step
    #lr_rate = tf.train.exponential_decay(
    #        FLAGS.learning_rate, global_step,
    #        decay_step, decay_rate, staircase=True)

    #momentum = 0.9
    # define the training paramters and model, 
    #train_step = tf.train.MomentumOptimizer(
    #        lr_rate, momentum, use_nesterov=True).minimize(
    #                loss, global_step=global_step) 



    # Create summaries to visualize weights
#    for var in tf.trainable_variables():
#        tf.histogram_summary(var.name, var)

    # Summarize all gradients
#    grads = tf.gradients(loss, tf.trainable_variables())
#    grads = list(zip(grads, tf.trainable_variables()))
#    for grad, var in grads:
#        tf.histogram_summary(var.name + '/gradient', grad)
    #####################################################################


    ######################################################################
    # Start training
    training_epoches = FLAGS.num_epochs
    batch_size = FLAGS.batch_size
    display_step = 20
    total_batches = X.shape[0] // batch_size
    num_epochs_save = FLAGS.num_epochs_save

    # saver to save and restore all variables
    saver = tf.train.Saver()

    # summary
    tf_version = tf.__version__.rpartition('.')[0]
    if parse_version(tf_version) >= parse_version('0.12.0'):
        summary_op = tf.summary.merge_all()
    else:
        summary_op = tf.merge_all_summaries()

    with tf.Session() as sess:
        if parse_version(tf_version) >= parse_version('0.12.0'):
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter(logs_path, 
                        graph=tf.get_default_graph())
        else:
            sess.run(tf.initialize_all_variables())
            writer = tf.train.SummaryWriter(logs_path, 
                        graph=tf.get_default_graph())
        for epoch in range(training_epoches):
            avg_cost = 0.
            batch_idx = 0
            for batch in  eeg.iterate_minibatches(batch_size, shuffle=True):
                batch_idx += 1
                batch_xs = batch
                feeds = {x: batch_xs, dropout_keep_prob: 0.5, is_training: True}
                _, step, summary = sess.run(
                        [train_step, global_step, summary_op],
                        feed_dict=feeds)

                # write log
                writer.add_summary(summary, step)

                train_loss = loss.eval({
                                        x: batch_xs, 
                                        dropout_keep_prob: 1.0, 
                                        is_training: False })
                avg_cost += train_loss
                if batch_idx % display_step == 0:
                    print('  step %6d, loss = %6.5f' % (batch_idx, train_loss))

            eval_loss = loss.eval({
                                    x: eeg._valid_data, 
                                    dropout_keep_prob: 1.0,
                                    is_training: False })
            l1_loss_network = l1_loss.eval({
                        x: eeg._valid_data, 
                        dropout_keep_prob: 1.0, 
                        is_training: False})
            current_step = tf.train.global_step(sess, global_step)
            avg_epoch_loss = avg_cost / total_batches
            print('Epoch {:6d}, step {:6d}, l1_loss= {:6.5f}, agv_loss= {:6.5f}, eval_los= {:6.5f}'.format(epoch, current_step, l1_loss_network, avg_epoch_loss, eval_loss))

            if (epoch+1) % num_epochs_save == 0:
                model_file_fullpath = model_file_prefix + str(epoch+1) + '_' + \
                    datetime.now().strftime('%Y-%m-%d-%H%M%S') + '.ckpt'
                # retrieve the current global_step
                #current_step = tf.train.global_step(sess, global_step)
                save_path = saver.save(
                                sess, model_file_fullpath, 
                                global_step=current_step)
                print("Model saved in file: %s" % save_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_base_dir', type=str, 
        default='/home/chuong/EEG-Project/processed_data',
        help='Directory for storing input data')
    parser.add_argument('--trained_model_base_dir', type=str, 
        default='/home/chuong/EEG-Project/trained_models',
        help='Output directory')
    parser.add_argument('--model', type=str, 
        default='big', help='Model size: big, small')
    parser.add_argument('--feature_activation', type=str, 
            default='relu', help='Activation function: relu, softmax')
    parser.add_argument('--data_type', type=str, 
        default='subsample', help='Subsampling (subsample, freqSum)')
    parser.add_argument('--num_epochs', type=int, default=50, 
        help='Number of epochs')
    parser.add_argument('--num_epochs_save', type=int, 
        default=10, help='Number epochs to trigger model saving to disk')
    parser.add_argument('--batch_size', type=int, 
        default=32, help='Mini-batch size')
    parser.add_argument('--learning_rate', type=float , 
        default=1e-6, help='Learning rate')
    #parser.add_argument('--decay_rate', type=float , 
    #    default=0.8, help='Decay rate')
    #parser.add_argument('--decay_step', type=float , 
    #    default=5000, help='Decay step')
    parser.add_argument('--gamma', type=float , 
        default=1e-7, help='Regularization gain')
    parser.add_argument('--test', type=bool, 
        default=False, help='True for fake data')
    parser.add_argument('--data_normalization', type=str, 
        default='scaling', help='Data normalization: scaling, normalize, none')

    FLAGS, unparsed = parser.parse_known_args()
    #tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    tf.app.run(main=main)
