# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
from tensorflow.python.ops.rnn_cell_impl import GRUCell
import numpy as np

# Linear layer
def linear(input_, output_size, scope=None):
    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    with tf.variable_scope(scope or "SimpleLinear"):
        matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)

    return tf.matmul(input_, tf.transpose(matrix)) + bias_term


# Highway network. Refer to http://arxiv.org/abs/1505.00387
def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(linear(input_, size, scope='highway_lin_%d' % idx))
            t = tf.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)
            output = t * g + (1. - t) * input_
            input_ = output
    return output


class Rewarder(object):
    def __init__(self, num_emb, batch_size, emb_dim, hidden_dim,
                 sequence_length, l2_reg_lambda=0):
        self.filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
        self.num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]

        self.vocab_size = num_emb
        self.batch_size = batch_size
        self.embedding_size = emb_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.r_params = []
        self.grad_clip = 5.0
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, ], name="input_y")
        self.dis_learning_rate = tf.placeholder(tf.float32, name="lr")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="drop_rate")
        self.l2_loss = tf.constant(0.0)

        with tf.variable_scope('rewarder'):
            # Embedding layer
            with tf.device('/cpu:0'), tf.name_scope("embedding"):
                self.W = tf.Variable(
                    tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
                    name="W")
                self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x) # (batch_size, sequence_length, embedding_size)

            # Encode the text with GRU
            cell_enc = GRUCell(self.hidden_dim)
            encoder_output, _ = tf.nn.dynamic_rnn(cell_enc, self.embedded_chars, dtype=tf.float32) # batch_size, sequence_length, hidden_dim
            self.embedded_chars_expanded = tf.expand_dims(encoder_output, -1)

            # Construct convolution and maxpool layer
            pooled_outputs = []
            for filter_size, num_filter in zip(self.filter_sizes, self.num_filters):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, self.hidden_dim, 1, num_filter]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filter]), name="b")
                    conv = tf.nn.conv2d(self.embedded_chars_expanded, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(h, ksize=[1, self.sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool")
                    pooled_outputs.append(pooled)

            # Combine all the pooled features
            num_filters_total = sum(self.num_filters)
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

            # Add highway
            with tf.name_scope("highway"):
                self.h_highway = highway(self.h_pool_flat, self.h_pool_flat.get_shape()[1], 1, 0)

            # Add dropout
            with tf.name_scope("dropout"):
                self.h_drop = tf.nn.dropout(self.h_highway, self.dropout_keep_prob)

            # Final scores
            with tf.name_scope("output"):
                W = tf.Variable(tf.truncated_normal([num_filters_total, 1], stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[1]), name="b")
                self.l2_loss += tf.nn.l2_loss(W)
                self.l2_loss += tf.nn.l2_loss(b)
                self.scores = tf.nn.sigmoid(tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")) # batch_size

            # Calculate least-square loss
            with tf.name_scope("loss"):
                self.labels = tf.reshape(self.input_y, [-1, 1])
                losses = tf.reduce_sum((self.scores - self.labels) * (self.scores - self.labels), 1)
                self.loss = tf.reduce_mean(losses) + l2_reg_lambda * self.l2_loss

        self.params = [param for param in tf.trainable_variables() if 'rewarder' in param.name]
        d_optimizer = tf.train.AdamOptimizer(self.dis_learning_rate)
        grads_and_vars = d_optimizer.compute_gradients(self.loss, self.params, aggregation_method=2)
        self.train_op = d_optimizer.apply_gradients(grads_and_vars)


    def reward_train_step(self, sess, x, y, lr_rate, drop_rate):
        outputs = sess.run([self.train_op, self.loss], feed_dict={self.input_x: x, self.input_y: y,
                                                                  self.dis_learning_rate: lr_rate, self.dropout_keep_prob:drop_rate})
        return outputs

    def get_reward(self, sess, x):
        outputs = sess.run([self.scores], feed_dict={self.input_x: x, self.dropout_keep_prob: 1.0})
        return outputs
