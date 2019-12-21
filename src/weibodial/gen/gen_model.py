import tensorflow as tf
import numpy as np
import my_seq2seq
import my_loss
import my_attention_decoder_fn

from tensorflow.python.ops.nn import dynamic_rnn
from tensorflow.python.ops.rnn_cell_impl import GRUCell, MultiRNNCell
from tensorflow.contrib.lookup.lookup_ops import HashTable, KeyValueTensorInitializer
from tensorflow.contrib.layers.python.layers import layers
from my_output_projection import output_projection_layer
from tensorflow.python.ops import variable_scope


PAD_ID = 0
UNK_ID = 1
GO_ID = 2
EOS_ID = 3
_START_VOCAB = ['_PAD', '_UNK', '_GO', '_EOS']

class gen_model(object):
    def __init__(self,
            num_symbols,
            num_embed_units,
            num_units,
            num_layers,
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
        self.weight = tf.placeholder(tf.float32, shape=[None]) # batch

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

        self.responses_target = self.symbol2index.lookup(self.responses)  # batch, len
        batch_size, decoder_len = tf.shape(self.responses)[0], tf.shape(self.responses)[1]
        self.responses_input = tf.concat([tf.ones([batch_size, 1], dtype=tf.int32)*GO_ID,
            tf.split(self.responses_target, [decoder_len-1, 1], 1)[0]], 1)   # batch, len
        self.decoder_mask = tf.reshape(tf.cumsum(tf.one_hot(self.responses_length-1,
            decoder_len), reverse=True, axis=1), [-1, decoder_len]) # batch, len

        self.decoder_input = tf.nn.embedding_lookup(self.embed, self.responses_input)

        # Construct multi-layer GRU cells for encoder and decoder
        cell_enc = MultiRNNCell([GRUCell(num_units) for _ in range(num_layers)])
        cell_dec = MultiRNNCell([GRUCell(num_units) for _ in range(num_layers)])

        # Encode the post sequence
        encoder_output, encoder_state = tf.nn.dynamic_rnn(cell_enc, self.encoder_input, self.posts_length, dtype=tf.float32, scope="encoder")

        output_fn, sampled_sequence_loss = output_projection_layer(num_units, num_symbols, num_samples)
        attention_keys, attention_values, attention_score_fn, attention_construct_fn \
            = my_attention_decoder_fn.prepare_attention(encoder_output, 'bahdanau', num_units)

        # Decode the response sequence (Training)
        with variable_scope.variable_scope('decoder'):
            decoder_fn_train = my_attention_decoder_fn.attention_decoder_fn_train(encoder_state, attention_keys,
                attention_values, attention_score_fn, attention_construct_fn)
            self.decoder_output, _, _ = my_seq2seq.dynamic_rnn_decoder(cell_dec, decoder_fn_train,
                                                            self.decoder_input, self.responses_length, scope = 'decoder_rnn')
            self.decoder_loss = my_loss.sequence_loss(self.decoder_output, self.responses_target, self.decoder_mask,
                                                      softmax_loss_function = sampled_sequence_loss)
            self.weighted_decoder_loss = self.decoder_loss * self.weight

        attention_keys_infer, attention_values_infer, attention_score_fn_infer, attention_construct_fn_infer \
            = my_attention_decoder_fn.prepare_attention(encoder_output, 'bahdanau', num_units, reuse = True)

        # Decode the response sequence (Inference)
        with variable_scope.variable_scope('decoder', reuse = True):
            decoder_fn_inference = my_attention_decoder_fn.attention_decoder_fn_inference(output_fn,
                                                                                       encoder_state,
                                                                                       attention_keys_infer,
                                                                                       attention_values_infer,
                                                                                       attention_score_fn_infer,
                                                                                       attention_construct_fn_infer,
                                                                                       self.embed, GO_ID, EOS_ID,
                                                                                       max_length, num_symbols)
            self.decoder_distribution, _, _ = my_seq2seq.dynamic_rnn_decoder(cell_dec, decoder_fn_inference, scope = 'decoder_rnn')
            self.generation_index = tf.argmax(tf.split(self.decoder_distribution,
                [2, num_symbols-2], 2)[1], 2) + 2 # for removing UNK
            self.generation = tf.nn.embedding_lookup(self.symbols, self.generation_index)

        self.params = [k for k in tf.trainable_variables() if name_scope in k.name]

        # initialize the training process
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)
        self.adv_global_step = tf.Variable(0, trainable=False)

        # calculate the gradient of parameters
        self.cost = tf.reduce_mean(self.weighted_decoder_loss)
        self.unweighted_cost = tf.reduce_mean(self.decoder_loss)
        opt = tf.train.AdamOptimizer(self.learning_rate)
        gradients = tf.gradients(self.cost, self.params)
        clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
        self.update = opt.apply_gradients(zip(clipped_gradients, self.params), global_step=self.global_step)

        all_variables = [k for k in tf.global_variables() if name_scope in k.name]
        self.saver = tf.train.Saver(all_variables, write_version=tf.train.SaverDef.V2,
                max_to_keep=5, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)
        self.adv_saver = tf.train.Saver(all_variables, write_version=tf.train.SaverDef.V2,
                max_to_keep=5, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

    def print_parameters(self):
        for item in self.params:
            print('%s: %s' % (item.name, item.get_shape()))

    # Training step
    def step(self, session, data, forward_only=False):
        input_feed = {self.posts: data['query'], self.posts_length: data['len_query'],
                      self.responses: data['ans'], self.responses_length: data['len_ans'], self.weight: data['weight']}
        if forward_only:
            output_feed = [self.cost, self.unweighted_cost]
        else:
            output_feed = [self.cost, self.unweighted_cost, self.gradient_norm, self.update]
        return session.run(output_feed, input_feed)

    # Inference process
    def inference(self, session, data):
        input_feed = {self.posts: data['query'], self.posts_length: data['len_query'],
                      self.responses: data['ans'], self.responses_length: data['len_ans'],
                      self.weight: data['weight']}
        output_feed = [self.generation]
        return session.run(output_feed, input_feed)

    # Acquire a batch of data for training / test
    def gen_train_batched_data(self, data):
        len_query = [len(p['query']) + 1 for p in data]
        len_ans = [len(p['ans']) + 1 for p in data]

        def padding(sent, l):
            return sent + ['_EOS'] + ['_PAD'] * (l - len(sent) - 1)

        batched_query = [padding(p['query'], max(len_query)) for p in data]
        batched_ans = [padding(p['ans'], max(len_ans)) for p in data]
        batched_weight = [p['weight'] for p in data]
        batched_data = {'query': np.array(batched_query),
                        'len_query': np.array(len_query, dtype=np.int32),
                        'ans': np.array(batched_ans),
                        'len_ans': np.array(len_ans, dtype=np.int32),
                        'weight': np.array(batched_weight)}
        return batched_data
