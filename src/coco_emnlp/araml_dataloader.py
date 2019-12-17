import numpy as np
import scipy
import os
from scipy.misc import comb
import math
import lm.lm_train as lm_train
import time

# Compute the distribution for edit distance as in the RAML paper
def generate_payoff_distribution(max_sent_len, vocab_size, tau=1.0):
    probs = dict()
    Z_qs = dict()
    for sent_len in range(1, max_sent_len + 1):
        counts = [1.]  # e = 0, count = 1
        for e in range(1, sent_len + 1):
            count = comb(sent_len, e) * math.exp(-e / tau) * ((vocab_size - 1) ** (e - e / tau))
            counts.append(count)
        Z_qs[sent_len] = Z_q = sum(counts)
        prob = [count / Z_q for count in counts]
        probs[sent_len] = prob

    return probs, Z_qs


class Gen_Data_loader():
    def __init__(self, batch_size, tau, vocab_size, sample_size):
        self.batch_size = batch_size * sample_size
        self.batch_size_augment = batch_size * sample_size
        self.token_stream = []
        self.token_len = []
        self.augment_token_stream = []
        self.tau = tau
        self.vocab_size = vocab_size
        self.sample_size = sample_size
        self.max_length = 0

    def create_batches(self, data_file, PAD_ID, SEQ_LENGTH):
        self.token_stream = []
        self.augment_token_stream = []
        max_length = 0
        with open(data_file, 'r') as f:
            for line in f:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == SEQ_LENGTH:
                    self.token_stream.append(parse_line)
                if PAD_ID in parse_line:
                    tgt_sent_len = parse_line.index(PAD_ID)
                else:
                    tgt_sent_len = len(parse_line)
                if tgt_sent_len > max_length:
                    max_length = tgt_sent_len
        self.max_length = max_length

        # Construct the original data
        self.num_batch = int(len(self.token_stream) / self.batch_size)
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        self.sequence_batch = np.split(np.array(self.token_stream), self.num_batch, 0)
        self.pointer = 0

    def create_batches_augment(self, raml_file, LM_model_fw, LM_model_bw, sess, PAD_ID):
        if os.path.exists(raml_file):
            # Load the data from cache
            input_file = open(raml_file, 'r')
            for line in input_file.readlines():
                self.augment_token_stream.append(line.strip().split())
        else:
            # Construct the sampled data from Ps
            payoff_prob, Z_qs = generate_payoff_distribution(self.max_length, vocab_size=self.vocab_size, tau=self.tau)
            idx = 0
            st_time = time.time()
            for sent in self.token_stream:
                tgt_sent = sent
                if PAD_ID in tgt_sent:
                    tgt_sent_len = tgt_sent.index(PAD_ID)
                else:
                    tgt_sent_len = len(tgt_sent)
                sampling_len = np.random.choice(range(tgt_sent_len + 1), p=payoff_prob[tgt_sent_len],
                                                size=self.sample_size, replace=True)
                if 0 not in sampling_len:
                    sampling_len[0] = 0

                for i, e in enumerate(sampling_len):
                    if e > 0:
                        # Sample positions
                        new_tgt_sent = list(tgt_sent)
                        if e > tgt_sent_len:
                            old_word_pos = np.random.choice(range(tgt_sent_len), size=e)
                        else:
                            old_word_pos = np.random.choice(range(tgt_sent_len), size=e, replace=False)

                        # Sample words
                        for j in old_word_pos:
                            fw_tgt_sent = tgt_sent[:j]
                            bw_tgt_sent = tgt_sent[j + 1:]

                            fw_distribution = lm_train.lm_test(LM_model_fw, sess, np.array([fw_tgt_sent]), 1, 'lm_model_fw')
                            bw_distribution = lm_train.lm_test(LM_model_bw, sess, np.array([bw_tgt_sent]), 1, 'lm_model_bw')

                            final_distribution = np.vstack((fw_distribution[0], bw_distribution[0]))
                            final_distribution = final_distribution.min(0)  # [vocab_size]

                            exp_final_distribution = np.exp(final_distribution)
                            value_topk = exp_final_distribution / np.sum(exp_final_distribution)
                            word_pos = np.random.choice(range(self.vocab_size), p=value_topk)
                            num_sampling = 0
                            while word_pos == PAD_ID or word_pos == new_tgt_sent[j]:
                                word_pos = np.random.choice(range(self.vocab_size), p=value_topk)
                                num_sampling += 1
                                if num_sampling > 5:
                                    break
                            new_tgt_sent[j] = word_pos
                    else:
                        new_tgt_sent = list(tgt_sent)

                    self.augment_token_stream.append(new_tgt_sent)

                if (idx % 1000 == 0):
                    ed_time = time.time()
                    print 'process: ', idx, 'time: ', ed_time - st_time
                    st_time = time.time()

                idx += 1

            output_file = open(raml_file, 'w')
            for sent in self.augment_token_stream:
                output_file.write(' '.join([str(x) for x in sent]) + '\n')
            output_file.close()

        self.num_batch_augment = int(len(self.augment_token_stream) / self.batch_size_augment)
        self.augment_token_stream = self.augment_token_stream[:self.num_batch_augment * self.batch_size_augment]
        self.sequence_batch_augment = np.split(np.array(self.augment_token_stream), self.num_batch_augment, 0)
        self.pointer_augment = 0

    def next_batch(self):
        ret = self.sequence_batch[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def next_batch_augment(self):
        ret = self.sequence_batch_augment[self.pointer_augment]
        self.pointer_augment = (self.pointer_augment + 1) % self.num_batch_augment
        return ret

    def reset_pointer(self):
        self.pointer = 0

    def reset_pointer_augment(self):
        self.pointer_augment = 0


class Dis_dataloader():
    def __init__(self, batch_size):
        # Real data : generated data = 1:1
        self.batch_size = batch_size // 2
        self.sentences = np.array([])

    def load_train_data(self, positive_file, negative_file, SEQ_LENGTH):
        # Load positive and negative data
        positive_examples = []
        negative_examples = []
        with open(positive_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                positive_examples.append(parse_line)
        with open(negative_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == SEQ_LENGTH:
                    negative_examples.append(parse_line)
        self.pos_sentences = np.array(positive_examples)
        self.neg_sentences = np.array(negative_examples)

        # Shuffle the data
        shuffle_pos_indices = np.random.permutation(np.arange(len(self.pos_sentences)))
        shuffle_neg_indices = np.random.permutation(np.arange(len(self.neg_sentences)))
        self.pos_sentences = self.pos_sentences[shuffle_pos_indices]
        self.neg_sentences = self.neg_sentences[shuffle_neg_indices]

        # Construct the label
        self.positive_labels = np.array([1.0 for _ in positive_examples])
        self.negative_labels = np.array([0.0 for _ in negative_examples])

        # Split batches
        self.num_batch = int(min(len(self.pos_sentences),len(self.neg_sentences)) / self.batch_size)
        self.pos_sentences = self.pos_sentences[:self.num_batch * self.batch_size]
        self.positive_labels = self.positive_labels[:self.num_batch * self.batch_size]
        self.neg_sentences = self.neg_sentences[:self.num_batch * self.batch_size]
        self.negative_labels = self.negative_labels[:self.num_batch * self.batch_size]
        self.pos_sentences_batches = np.split(self.pos_sentences, self.num_batch, 0)
        self.positive_labels_batches = np.split(self.positive_labels, self.num_batch, 0)
        self.neg_sentences_batches = np.split(self.neg_sentences, self.num_batch, 0)
        self.negative_labels_batches = np.split(self.negative_labels, self.num_batch, 0)
        self.pointer = 0

    def next_batch(self):
        texts = np.concatenate((self.pos_sentences_batches[self.pointer], self.neg_sentences_batches[self.pointer]), axis=0)
        labels = np.concatenate((self.positive_labels_batches[self.pointer], self.negative_labels_batches[self.pointer]), axis=0)
        self.pointer = (self.pointer + 1) % self.num_batch
        return texts, labels

    def reset_pointer(self):
        self.pointer = 0
