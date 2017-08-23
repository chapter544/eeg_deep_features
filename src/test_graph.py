import tensorflow as tf
from models.fc_freqComponent_Models import build_fc_layer

x_dim = 100
with tf.name_scope("input"):
    x = tf.placeholder(tf.float32, [None, x_dim], name='x')
    dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_rate')
    is_training = tf.placeholder(tf.bool, name='training_phase')

fc1 = build_fc_layer(x, x_dim, 50, "FC1", is_training)
fc2 = build_fc_layer(fc1, 50, 20, "FC2", is_training)

with tf.Session() as sess:
    writer = tf.summary.FileWriter("myLogs", graph=tf.get_default_graph())

