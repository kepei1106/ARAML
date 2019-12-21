import tensorflow as tf
import numpy as np

from tensorflow.python.ops.nn import dynamic_rnn
from tensorflow.python.ops.rnn_cell_impl import GRUCell
from tensorflow.contrib.lookup.lookup_ops import HashTable, KeyValueTensorInitializer
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.ops import variable_scope


PAD_ID = 0
UNK_ID = 1
GO_ID = 2
EOS_ID = 3
_START_VOCAB = ['_PAD', '_UNK', '_GO', '_EOS']

class disc_model(object):
    def __init__(self,
            num_symbols,
            num_embed_units,
            num_units,
            vocab=None,
            embed=None,
            name_scope = None,
            learning_rate=0.0001,
            learning_rate_decay_factor=0.95,
            max_gradient_norm=5,
            l2_lambda = 0.2):

        self.posts = tf.placeholder(tf.string, shape=[None, None])  # batch * len
        self.posts_length = tf.placeholder(tf.int32, shape=[None])  # batch
        self.responses = tf.placeholder(tf.string, shape=[None, None])  # batch*len
        self.responses_length = tf.placeholder(tf.int32, shape=[None])  # batch
        self.generation = tf.placeholder(tf.string, shape=[None, None])  # batch*len
        self.generation_length = tf.placeholder(tf.int32, shape=[None])  # batch

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
        self.posts_input_embed = tf.nn.embedding_lookup(self.embed, self.posts_input) #batch * utter_len * embed_unit
        self.responses_input = self.symbol2index.lookup(self.responses)
        self.responses_input_embed = tf.nn.embedding_lookup(self.embed, self.responses_input) # batch * utter_len * embed_unit
        self.generation_input = self.symbol2index.lookup(self.generation)
        self.generation_input_embed = tf.nn.embedding_lookup(self.embed, self.generation_input) # batch * utter_len * embed_unit

        # Construct bidirectional GRU cells for encoder / decoder
        cell_fw_post = GRUCell(num_units)
        cell_bw_post = GRUCell(num_units)
        cell_fw_resp = GRUCell(num_units)
        cell_bw_resp = GRUCell(num_units)

        # Encode the post sequence
        with variable_scope.variable_scope("post_encoder"):
            posts_state, posts_final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw_post, cell_bw_post, self.posts_input_embed,
                                                                              self.posts_length, dtype=tf.float32)
            posts_final_state_bid = tf.concat(posts_final_state, 1)  # batch_size * (2 * num_units)

        # Encode the real response sequence
        with variable_scope.variable_scope("resp_encoder"):
            responses_state, responses_final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw_resp, cell_bw_resp, self.responses_input_embed,
                                                                              self.responses_length, dtype=tf.float32)
            responses_final_state_bid = tf.concat(responses_final_state, 1)

        # Encode the generated response sequence
        with variable_scope.variable_scope("resp_encoder", reuse = True):
            generation_state, generation_final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw_resp, cell_bw_resp, self.generation_input_embed,
                                                                              self.generation_length, dtype=tf.float32)
            generation_final_state_bid = tf.concat(generation_final_state, 1)

        # Calculate the relevance score between post and real response
        with variable_scope.variable_scope("calibration"):
            self.W = tf.get_variable('W', [2 * num_units, 2 * num_units], tf.float32)
            vec_post = tf.reshape(posts_final_state_bid, [-1, 1, 2 * num_units])
            vec_resp = tf.reshape(responses_final_state_bid, [-1, 2 * num_units, 1])
            attn_score_true = tf.einsum('aij,ajk->aik', tf.einsum('aij,jk->aik', vec_post, self.W), vec_resp)
            attn_score_true = tf.reshape(attn_score_true, [-1, 1])
            fc_true_input = tf.concat([posts_final_state_bid, responses_final_state_bid, attn_score_true], 1)

            self.output_fc_W = tf.get_variable("output_fc_W", [4 * num_units + 1, num_units], tf.float32)
            self.output_fc_b = tf.get_variable("output_fc_b", [num_units], tf.float32)
            fc_true = tf.nn.tanh(tf.nn.xw_plus_b(fc_true_input, self.output_fc_W, self.output_fc_b))  # batch_size

            self.output_W = tf.get_variable("output_W", [num_units, 1], tf.float32)
            self.output_b = tf.get_variable("output_b", [1], tf.float32)
            self.cost_true = tf.nn.sigmoid(tf.nn.xw_plus_b(fc_true, self.output_W, self.output_b))  # batch_size

        # Calculate the relevance score between post and generated response
        with variable_scope.variable_scope("calibration", reuse = True):
            vec_gen = tf.reshape(generation_final_state_bid, [-1, 2 * num_units, 1])
            attn_score_false = tf.einsum('aij,ajk->aik', tf.einsum('aij,jk->aik', vec_post, self.W), vec_gen)
            attn_score_false = tf.reshape(attn_score_false, [-1, 1])
            fc_false_input = tf.concat([posts_final_state_bid, generation_final_state_bid, attn_score_false], 1)
            fc_false = tf.nn.tanh(tf.nn.xw_plus_b(fc_false_input, self.output_fc_W, self.output_fc_b))  # batch_size
            self.cost_false = tf.nn.sigmoid(tf.nn.xw_plus_b(fc_false, self.output_W, self.output_b))  # batch_size

        self.PR_cost = tf.reduce_mean(tf.reduce_sum(tf.square(self.cost_true - 1.0), axis = 1))
        self.PG_cost = tf.reduce_mean(tf.reduce_sum(tf.square(self.cost_false), axis = 1))

        # Use the loss similar to least square GAN
        self.cost = self.PR_cost / 2.0 + self.PG_cost / 2.0 + l2_lambda * (
            tf.nn.l2_loss(self.output_fc_W) + tf.nn.l2_loss(self.output_fc_b) +
            tf.nn.l2_loss(self.output_W) + tf.nn.l2_loss(self.output_b) + tf.nn.l2_loss(self.W))

        # building graph finished and get all parameters
        self.params = [k for k in tf.trainable_variables() if name_scope in k.name]

        # initialize the training process
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)
        self.adv_global_step = tf.Variable(0, trainable=False)

        # calculate the gradient of parameters
        opt = tf.train.AdamOptimizer(self.learning_rate)
        gradients = tf.gradients(self.cost, self.params)
        clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
        self.update = opt.apply_gradients(zip(clipped_gradients, self.params), global_step=self.global_step)
        self.reward = tf.reduce_sum(self.cost_false, axis = 1) # batch


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
                      self.responses: data['ans'], self.responses_length: data['len_ans'],
                      self.generation: data['gen'], self.generation_length: data['len_gen']}
        if forward_only:
            output_feed = [self.cost]
        else:
            output_feed = [self.cost, self.gradient_norm, self.update]
        return session.run(output_feed, input_feed)

    # Inference step
    def inference(self, session, data):
        input_feed = {self.posts: data['query'], self.posts_length: data['len_query'],
                      self.responses: data['ans'], self.responses_length: data['len_ans'],
                      self.generation: data['ans'], self.generation_length: data['len_ans']}
        output_feed = [self.reward]
        return session.run(output_feed, input_feed)

    # Acquire a batch of data during inference
    def gen_test_batched_data(self, infer_data):
        len_query = [len(p['query']) + 1 for p in infer_data]
        len_ans = [len(p['ans']) + 1 for p in infer_data]

        def padding(sent, l):
            return sent + ['_EOS'] + ['_PAD'] * (l - len(sent) - 1)

        batched_query = [padding(p['query'], max(len_query)) for p in infer_data]
        batched_ans = [padding(p['ans'], max(len_ans)) for p in infer_data]
        batched_data = {'query': np.array(batched_query),
                        'len_query': np.array(len_query, dtype=np.int32),
                        'ans': np.array(batched_ans),
                        'len_ans': np.array(len_ans, dtype=np.int32)}
        return batched_data

    # Acquire a batch of data during training
    def gen_train_batched_data(self, data):
        len_query = [len(p['query']) + 1 for p in data]
        len_ans = [len(p['ans']) + 1 for p in data]
        len_gen = [len(p['gen']) + 1 for p in data]

        def padding(sent, l):
            return sent + ['_EOS'] + ['_PAD'] * (l - len(sent) - 1)

        batched_query = [padding(p['query'], max(len_query)) for p in data]
        batched_ans = [padding(p['ans'], max(len_ans)) for p in data]
        batched_gen = [padding(p['gen'], max(len_gen)) for p in data]
        batched_data = {'query': np.array(batched_query),
                        'len_query': np.array(len_query, dtype=np.int32),
                        'ans': np.array(batched_ans),
                        'len_ans': np.array(len_ans, dtype=np.int32),
                        'gen': np.array(batched_gen),
                        'len_gen': np.array(len_gen, dtype=np.int32)}
        return batched_data
