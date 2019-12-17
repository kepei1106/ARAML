import tensorflow as tf
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.ops import variable_scope

def output_projection_layer(num_units, num_symbols, num_samples, name="output_projection"):
    # Project the hidden vector to the probability on the vocabulary
    def output_fn(outputs):
        with variable_scope.variable_scope(name, reuse = True):
            l_fc1 = tf.reshape(outputs, [-1, num_units]) # batch * len
            weights = tf.transpose(tf.get_variable("weights", [num_units, num_symbols]))
            bias = tf.get_variable("biases", [num_symbols])
            l_fc2 = tf.matmul(l_fc1, tf.transpose(weights)) + bias
            y_dis = tf.nn.softmax(l_fc2)
        return y_dis

    # Calculate the cross entropy loss
    def my_sequence_loss(outputs, targets):
        with variable_scope.variable_scope('decoder_rnn/%s' % name):
            weights = tf.transpose(tf.get_variable("weights", [num_units, num_symbols]))
            bias = tf.get_variable("biases", [num_symbols])
            local_labels = tf.reshape(targets, [-1, 1])
            local_outputs = tf.reshape(outputs, [-1, num_units])
            logits = tf.matmul(local_outputs, tf.transpose(weights)) + bias
            cross_entropy = tf.nn.sampled_softmax_loss(weights, bias, local_labels,
                                                    local_outputs, num_samples, num_symbols)

        return cross_entropy, logits
    
    return output_fn, my_sequence_loss
