import tensorflow as tf
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.ops import variable_scope

def output_projection_layer(num_units, num_symbols, num_samples, name="output_projection"):
    def output_fn(outputs):
        with variable_scope.variable_scope(name, reuse = True):
            l_fc1 = tf.reshape(outputs, [-1, num_units]) # batch * len
            weights = tf.transpose(tf.get_variable("weights", [num_units, num_symbols]))
            bias = tf.get_variable("biases", [num_symbols])
            l_fc2 = tf.matmul(l_fc1, tf.transpose(weights)) + bias
            y_dis = tf.nn.softmax(l_fc2)
        return y_dis


    def my_sequence_loss(outputs, targets):
        with variable_scope.variable_scope('decoder_rnn/%s' % name):
            weights = tf.transpose(tf.get_variable("weights", [num_units, num_symbols]))
            bias = tf.get_variable("biases", [num_symbols])

            local_labels = tf.reshape(targets, [-1]) # batch_size * len
            local_outputs = tf.reshape(outputs, [-1, num_units]) # batch_size * len, num_units
            local_dis = tf.nn.log_softmax(tf.matmul(local_outputs, tf.transpose(weights)) + bias) # batch * len, num_symbols
            labels_onehot = tf.one_hot(local_labels, num_symbols)
            labels_onehot = tf.clip_by_value(labels_onehot, 0.0, 1.0)
            cross_entropy = tf.reduce_sum(-labels_onehot * local_dis, 1) # batch * len

        return cross_entropy
    
    return output_fn, my_sequence_loss
    
