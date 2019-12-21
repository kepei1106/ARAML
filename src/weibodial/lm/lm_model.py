import tensorflow as tf
import numpy as np
import my_seq2seq
import my_loss
import my_simple_decoder_fn

from tensorflow.python.ops.nn import dynamic_rnn
from tensorflow.python.ops.rnn_cell_impl import GRUCell
from tensorflow.contrib.lookup.lookup_ops import HashTable, KeyValueTensorInitializer
from tensorflow.contrib.layers.python.layers import layers
from my_output_projection import output_projection_layer
from tensorflow.python.ops import variable_scope

PAD_ID = 0
UNK_ID = 1
GO_ID = 2
EOS_ID = 3
_START_VOCAB = ['_PAD', '_UNK', '_GO', '_EOS']

class lm_model(object):
    def __init__(self,
            num_symbols,
            num_embed_units,
            num_units,
            vocab=None,
            embed=None,
            name_scope = None,
            learning_rate=0.001,
            learning_rate_decay_factor=0.95,
            max_gradient_norm=5,
            num_samples=512,
            max_length=30):

        self.posts = tf.placeholder(tf.string, shape=[None, None])  # batch * len
        self.posts_length = tf.placeholder(tf.int32, shape=[None])  # batch
        self.responses = tf.placeholder(tf.string, shape=[None, None])  # batch*len
        self.responses_length = tf.placeholder(tf.int32, shape=[None])  # batch

        # build the vocab table (string to index)
        self.symbols = tf.Variable(vocab, trainable=False, name="symbols")
        self.symbol2index = HashTable(KeyValueTensorInitializer(self.symbols, 
            tf.Variable(np.array([i for i in range(num_symbols)], dtype=np.int32), False)), 
            default_value=UNK_ID, name="symbol2index")

        # build the embedding table (index to vector)
        if embed is None:
            # initialize the embedding randomly
            self.embed = tf.get_variable('embed', [num_symbols, num_embed_units], tf.float32)
        else:
            # initialize the embedding by pre-trained word vectors
            self.embed = tf.get_variable('embed', dtype=tf.float32, initializer=embed)

        self.posts_input = self.symbol2index.lookup(self.posts)   # batch * utter_len
        self.encoder_input = tf.nn.embedding_lookup(self.embed, self.posts_input) # batch * utter_len * embed_unit

        self.responses_target = self.symbol2index.lookup(self.responses)  # batch*len
        batch_size, decoder_len = tf.shape(self.responses)[0], tf.shape(self.responses)[1]
        self.responses_input = tf.concat([tf.ones([batch_size, 1], dtype=tf.int32)*GO_ID,
            tf.split(self.responses_target, [decoder_len-1, 1], 1)[0]], 1)   # batch*len
        self.decoder_mask = tf.reshape(tf.cumsum(tf.one_hot(self.responses_length-1,
            decoder_len), reverse=True, axis=1), [-1, decoder_len]) # batch * len

        self.decoder_input = tf.nn.embedding_lookup(self.embed, self.responses_input)

        # Construct GRU cells for encoder / decoder
        cell_enc = GRUCell(num_units)
        cell_dec = GRUCell(num_units)

        # Encode the post
        _, encoder_state = tf.nn.dynamic_rnn(cell_enc, self.encoder_input, self.posts_length, dtype = tf.float32, scope = "encoder")

        output_fn, sampled_sequence_loss = output_projection_layer(num_units, num_symbols, num_samples)

        # Decode the response (training phase)
        with variable_scope.variable_scope('decoder'):
            decoder_fn_train = my_simple_decoder_fn.simple_decoder_fn_train(encoder_state)
            self.decoder_output, _, _ = my_seq2seq.dynamic_rnn_decoder(cell_dec, decoder_fn_train,
                                                            self.decoder_input, self.responses_length, scope = "decoder_rnn")
            self.decoder_loss, self.all_decoder_output = my_loss.sequence_loss(self.decoder_output, self.responses_target, self.decoder_mask,
                                                      softmax_loss_function = sampled_sequence_loss)

        # Decode the response (inference phase)
        with variable_scope.variable_scope('decoder', reuse = True):
            decoder_fn_inference = my_simple_decoder_fn.simple_decoder_fn_inference(output_fn, encoder_state,
                                                                                    self.embed, GO_ID, EOS_ID,
                                                                                    max_length, num_symbols)
            self.decoder_distribution, _, _ = my_seq2seq.dynamic_rnn_decoder(cell_dec, decoder_fn_inference, scope = "decoder_rnn")
            self.generation_index = tf.argmax(tf.split(self.decoder_distribution, [2, num_symbols-2], 2)[1], 2) + 2 # for removing UNK
            self.generation = tf.nn.embedding_lookup(self.symbols, self.generation_index)

        self.params = [k for k in tf.trainable_variables() if name_scope in k.name]

        # Initialize the training process
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)

        # Calculate the gradient of parameters
        self.cost = tf.reduce_mean(self.decoder_loss)
        opt = tf.train.AdamOptimizer(self.learning_rate)
        gradients = tf.gradients(self.cost, self.params)
        clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
        self.update = opt.apply_gradients(zip(clipped_gradients, self.params), global_step=self.global_step)

        all_variables = [k for k in tf.global_variables() if name_scope in k.name]
        self.saver = tf.train.Saver(all_variables, write_version=tf.train.SaverDef.V2,
                max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

    def print_parameters(self):
        for item in self.params:
            print('%s: %s' % (item.name, item.get_shape()))

    # Conduct a step of training
    def step(self, session, data, forward_only=False):
        input_feed = {self.posts: data['query'], self.posts_length: data['len_query'],
                      self.responses: data['ans'], self.responses_length: data['len_ans']}
        if forward_only:
            output_feed = [self.cost]
        else:
            output_feed = [self.cost, self.gradient_norm, self.update]
        return session.run(output_feed, input_feed)

    # Get the language model score during inference
    def inference(self, session, data):
        input_feed = {self.posts: data['query'], self.posts_length: data['len_query'],
                      self.responses: data['ans'], self.responses_length: data['len_ans']}
        output_feed = [self.all_decoder_output]
        return session.run(output_feed, input_feed)

    # Acquire a batch of data used for training / test
    def gen_train_batched_data(self, data, config):
        len_query = [len(p['query']) + 1 for p in data]
        len_ans = [len(p['ans']) + 1 for p in data]

        def padding(sent, l, is_query = False):
            if config.direction == 0 and is_query == False:
                sent.reverse()
            return sent + ['_EOS'] + ['_PAD'] * (l - len(sent) - 1)

        batched_query = [padding(p['query'], max(len_query), True) for p in data]
        batched_ans = [padding(p['ans'], max(len_ans)) for p in data]
        batched_data = {'query': np.array(batched_query),
                        'len_query': np.array(len_query, dtype=np.int32),
                        'ans': np.array(batched_ans),
                        'len_ans': np.array(len_ans, dtype=np.int32)}
        return batched_data
