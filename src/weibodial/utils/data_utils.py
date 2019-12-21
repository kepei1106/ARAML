import numpy as np
import tensorflow as tf
import sys
import time
import os
import math
from scipy.misc import comb

sys.path.append("..")

_START_VOCAB = ['_PAD', '_UNK', '_GO', '_EOS']

# Load the sampled data from the cache
def load_raml_data(file_path):
    with open('%s/raml_post.txt' % file_path) as f:
        train_query = [line.strip().split() for line in f.readlines()]
    with open('%s/raml_resp.txt' % file_path) as f:
        train_ans = [line.strip().split() for line in f.readlines()]
    raml_train = []
    for p, r in zip(train_query, train_ans):
        raml_train.append({'query': p, 'ans': r, 'weight': 1.0})
    return raml_train

# Save the sampled data from Ps
def save_raml_data(raml_data, file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    f1 = open('%s/raml_post.txt' % file_path, 'w')
    f2 = open('%s/raml_resp.txt' % file_path, 'w')
    for pair in raml_data:
        f1.write(' '.join(pair['query']) + '\n')
        f2.write(' '.join(pair['ans']) + '\n')
    f1.close()
    f2.close()

# Load the data to train the discriminator
def load_disc_data(file_path):
    with open('%s/post_samples.txt' % file_path) as f:
        post_samples = [line.strip().split() for line in f.readlines()]
    with open('%s/positive_samples.txt' % file_path) as f:
        positive_samples = [line.strip().split() for line in f.readlines()]
    with open('%s/negative_samples.txt' % file_path) as f:
        negative_samples = [line.strip().split() for line in f.readlines()]
    data_train = []
    for p, r, g in zip(post_samples, positive_samples, negative_samples):
        data_train.append({'query': p, 'ans': r, 'gen':g})
    return data_train

# Load the training / test set
def load_gen_data(file_path):
    with open('%s/train.query' % file_path) as f:
        train_query = [line.strip().split() for line in f.readlines()]
    with open('%s/test.query' % file_path) as f:
        test_query = [line.strip().split() for line in f.readlines()]
    with open('%s/train.ans' % file_path) as f:
        train_ans = [line.strip().split() for line in f.readlines()]
    with open('%s/test.ans' % file_path) as f:
        test_ans = [line.strip().split() for line in f.readlines()]
    data_train, data_test = [], []
    for p, r in zip(train_query, train_ans):
        data_train.append({'query': p, 'ans': r})
    for p, r in zip(test_query, test_ans):
        data_test.append({'query': p, 'ans': r})
    return data_train, data_test

# Initialize the word embedding
def initialize_word2vec(vocab_list, vectors, embed_units):
    print("Initialize word vectors...")
    embed = []
    for word in vocab_list:
        if word in vectors:
            vector = map(float, vectors[word].split())
        else:
            vector = np.zeros((embed_units), dtype=np.float32)
        embed.append(vector)
    embed = np.array(embed, dtype=np.float32)
    return embed

# Load the pre-trained word vector (if exists)
def build_word2vec(path):
    print("Build word vectors...")
    vectors = {}
    if os.path.exists('%s/vector.txt' % path):
        with open('%s/vector.txt' % path) as f:
            for i, line in enumerate(f):
                if i % 100000 == 0:
                    print("    processing line %d" % i)
                s = line.strip()
                word = s[:s.find(' ')]
                vector = s[s.find(' ')+1:]
                vectors[word] = vector
    return vectors


# Build the vocabulary from training data
def build_vocab(data_train, vocab_size):
    print("Creating vocabulary...")
    vocab = {}
    for i, pair in enumerate(data_train):
        for token in pair['query'] + pair['ans']:
            if token in vocab:
                vocab[token] += 1
            else:
                vocab[token] = 1

    vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)

    print 'len(vocab_list) = ', len(vocab_list)

    if len(vocab_list) > vocab_size:
        vocab_list = vocab_list[:vocab_size] # remove words with low frequency from vocab_list

    return vocab_list


# Compute the payoff distribution based on edit distance (substitution only) as in the RAML paper
def generate_payoff_distribution(max_sent_len, vocab_size, tau=1.0):
    probs = dict()
    Z_qs = dict()
    for sent_len in range(1, max_sent_len + 1):
        counts = [1.]  # e = 0, count = 1
        for e in range(1, sent_len + 1):
            # apply the rescaling trick as in https://gist.github.com/norouzi/8c4d244922fa052fa8ec18d8af52d366
            count = comb(sent_len, e) * math.exp(-e / tau) * ((vocab_size - 1) ** (e - e / tau))
            counts.append(count)
        Z_qs[sent_len] = Z_q = sum(counts)
        prob = [count / Z_q for count in counts]
        probs[sent_len] = prob

    return probs, Z_qs