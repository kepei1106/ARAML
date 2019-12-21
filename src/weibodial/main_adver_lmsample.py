import os

import tensorflow as tf
import numpy as np
import sys
import time
import gen.gen_train as gen_train
import gen.gen_model as gen_model
import disc.disc_train as disc_train
import disc.disc_model as disc_model
import random
import utils.conf as conf
import utils.data_utils as data_utils
import lm.lm_train as lm_train
from utils.data_utils import generate_payoff_distribution
import math
import os

SEED = int(time.time())
random.seed(SEED)
np.random.seed(SEED)

gen_config = conf.gen_config
disc_config = conf.disc_config
lm_config_fw = conf.lm_forward_config
lm_config_bw = conf.lm_backward_config
adv_config = conf.adv_config

tf.app.flags.DEFINE_integer("sample_size", 5, "Number of samples before training.")
tf.app.flags.DEFINE_float("tau", 0.95, "Parameter of payoff distribution.")
tf.app.flags.DEFINE_float("alpha", 0.2, "Parameter of payoff distribution.")
tf.app.flags.DEFINE_boolean("is_train", True, "Set to False to inference.")

FLAGS = tf.app.flags.FLAGS

def read_dialog(file_path):
    data_train, data_test = data_utils.load_gen_data(file_path)
    # Initialize the weight of each sample
    for data in [data_train, data_test]:
        for idx in range(len(data)):
            data[idx]['weight'] = 1.0
    return data_train, data_test

def prepare_origin_data(config):
    # Load the training / test set
    gen_train, gen_test = read_dialog(config.data_dir)
    gen_data_list = [gen_train, gen_test]
    # Load the word vectors
    vectors = data_utils.build_word2vec(config.vector_dir)
    # Build the vocabulary
    gen_vocab_list = data_utils.build_vocab(gen_data_list[0], config.vocab_size)
    # Initialize the word embedding
    gen_embed = data_utils.initialize_word2vec(gen_vocab_list, vectors, config.embed_units)

    return gen_data_list, gen_vocab_list, gen_embed, vectors


def sampling(data_list, gen_vocab_list, gen_embed, gpu_config):
    data_train = data_list[0]

    # Build the payoff distribution based on edit distance
    print('build payoff distribution based on edit distance')
    payoff_prob, Z_qs = generate_payoff_distribution(max([len(sent['ans']) for sent in data_train]),
                                                    vocab_size=len(gen_vocab_list) - 4, tau=FLAGS.tau)

    with tf.Session(config = gpu_config) as sess:
        # Pre-train forward / backward language model
        LM_model_fw = lm_train.create_model(sess, lm_config_fw, gen_vocab_list, gen_embed)
        LM_model_bw = lm_train.create_model(sess, lm_config_bw, gen_vocab_list, gen_embed)
        lm_train.lm_train(LM_model_fw, sess, lm_config_fw, data_list)
        lm_train.lm_train(LM_model_bw, sess, lm_config_bw, data_list)

        # Get samples from Ps
        raml_data = []
        st_time = time.time()
        for idx in range(len(data_train)):
            tgt_samples = []
            src_sent = data_train[idx]['query']
            tgt_sent = data_train[idx]['ans']
            tgt_sent_len = len(data_train[idx]['ans'])
            # Sample the edit distance
            sampling_len = np.random.choice(range(tgt_sent_len + 1), p=payoff_prob[tgt_sent_len],
                                            size=FLAGS.sample_size, replace=True)
            if 0 not in sampling_len:
                sampling_len[0] = 0
            for i, e in enumerate(sampling_len):
                if e > 0:
                    # Sample positions
                    new_tgt_sent = list(tgt_sent)
                    if e > tgt_sent_len -1:
                        old_word_pos = np.random.choice(range(1, tgt_sent_len), size=e)
                    else:
                        old_word_pos = np.random.choice(range(1, tgt_sent_len), size=e, replace=False)
                    for j in old_word_pos:
                        # Sample words based on language model score
                        fw_tgt_sent = tgt_sent[:j]
                        bw_tgt_sent = tgt_sent[j+1:]
                        fw_distribution = lm_train.lm_test(LM_model_fw, sess, [{'query':src_sent, 'ans':fw_tgt_sent}], 1, lm_config_fw)
                        bw_distribution = lm_train.lm_test(LM_model_bw, sess, [{'query':src_sent, 'ans':bw_tgt_sent}], 1, lm_config_bw)
                        final_distribution = np.vstack((fw_distribution[0], bw_distribution[0]))
                        final_distribution = final_distribution.min(0) # [vocab_size]
                        exp_final_distribution = np.exp(final_distribution)
                        value_topk = exp_final_distribution / np.sum(exp_final_distribution)
                        word_pos = np.random.choice(range(len(gen_vocab_list)), p = value_topk)
                        num_sampling = 0
                        while gen_vocab_list[word_pos] == '_UNK' or gen_vocab_list[word_pos] == new_tgt_sent[j]:
                            word_pos = np.random.choice(range(len(gen_vocab_list)), p=value_topk)
                            num_sampling += 1
                            if num_sampling > 5:
                                break
                        new_tgt_sent[j] = gen_vocab_list[word_pos]
                else:
                    new_tgt_sent = list(tgt_sent)

                if '_EOS' in new_tgt_sent:
                    new_tgt_sent = new_tgt_sent[:new_tgt_sent.index('_EOS')]
                tgt_samples.append(new_tgt_sent)
                raml_data.append({'query': data_train[idx]['query'], 'ans': new_tgt_sent, 'weight': 1.0})

            if (idx % 1000 == 0):
                ed_time = time.time()
                print 'process: ', idx, 'time: ', ed_time - st_time
                st_time = time.time()
    return raml_data

def al_train(gpu_config, gen_config, disc_config, gen_vocab_list, gen_embed, gen_data_list, vectors):
    # Acquire samples from Ps
    if os.path.exists(gen_config.raml_data_dir + '/raml_post.txt'):
        raml_data = data_utils.load_raml_data(gen_config.raml_data_dir)
    else:
        raml_data = sampling(gen_data_list, gen_vocab_list, gen_embed, gpu_config)
        data_utils.save_raml_data(raml_data, gen_config.raml_data_dir)
    print('pre-sampling complete, len(raml_data) = ', len(raml_data))

    with tf.Session(config = gpu_config) as sess:
        # Create the model for generator
        al_gen_model = gen_train.create_model(sess, gen_config, gen_vocab_list, gen_embed, gen_config.train_dir)
        if FLAGS.is_train == False:
            gen_train.gen_test(sess, al_gen_model, gen_data_list[2], gen_config.test_dir, gen_config.batch_size, vectors)
            return

        # Pre-train generator
        gen_train.gen_pretrain(al_gen_model, sess, gen_config, gen_data_list[0])

        # Pre-train discriminator
        gen_train.generate_negative_samples(sess, al_gen_model, disc_config.first_sample_num, gen_data_list[0],
                                            disc_config.data_dir, gen_config.batch_size)
        disc_data = data_utils.load_disc_data(disc_config.data_dir)
        al_disc_model = disc_train.create_model(sess, disc_config, gen_vocab_list, gen_embed)
        disc_train.disc_train(al_disc_model, sess, disc_config, disc_data)

        disc_loss, gen_loss = 0.0, 0.0
        num_raml_batches = len(raml_data) // (adv_config.batch_size * FLAGS.sample_size)
        num_dis_batches = disc_config.sample_num // disc_config.batch_size
        gen_data_pointer = 0

        print("start adversarial training")
        print 'num_raml_batches = ', num_raml_batches
        for batch_id in range(adv_config.total_batch):
            # Train generator on the sampled data from Ps with weighted MLE objective
            for iter_gen in range(adv_config.gen_update_time):
                current_batch = raml_data[gen_data_pointer*adv_config.batch_size*FLAGS.sample_size: \
                                          (gen_data_pointer+1)*adv_config.batch_size*FLAGS.sample_size]
                gen_data_pointer += 1
                if gen_data_pointer == num_raml_batches - 1:
                    gen_data_pointer = 0
                batched_raml_data = al_gen_model.gen_train_batched_data(current_batch)
                disc_weight = al_disc_model.inference(sess, batched_raml_data)[0]
                total_weight = []
                for sample in range(adv_config.batch_size):
                    batch_weight = disc_weight[sample * FLAGS.sample_size:(sample + 1) * FLAGS.sample_size]
                    batch_weight = [math.exp(batch_weight[id] / FLAGS.alpha) for id in range(FLAGS.sample_size)]
                    normalizer = sum(batch_weight)
                    normalize_weight = np.array([w / normalizer for w in batch_weight])
                    total_weight.extend(normalize_weight)
                batched_raml_data['weight'] = np.array(total_weight)
                gen_loss += al_gen_model.step(sess, batched_raml_data)[0] * FLAGS.sample_size / adv_config.gen_update_time

            # Train discriminator to distinguish real data and generated data
            for iter_d in range(adv_config.dis_update_time):
                gen_train.generate_negative_samples(sess, al_gen_model, disc_config.sample_num, gen_data_list[0],
                                                    disc_config.data_dir, gen_config.batch_size)
                disc_data_adv = data_utils.load_disc_data(disc_config.data_dir)
                for iter_dis in range(adv_config.dis_epoch_per_update):
                    for batch_dis in range(num_dis_batches):
                        current_batch = disc_data_adv[batch_dis * disc_config.batch_size:(batch_dis + 1) * disc_config.batch_size]
                        batched_disc_data = al_disc_model.gen_train_batched_data(current_batch)
                        disc_loss += al_disc_model.step(sess, batched_disc_data)[0] / (adv_config.dis_epoch_per_update * num_dis_batches)

            # Test: generate responses to the posts in the test set
            if batch_id % adv_config.test_per_checkpoint == 0:
                gen_train.gen_test(sess, al_gen_model, gen_data_list[1], gen_config.test_dir, gen_config.batch_size, batch_id)

            # Save model parameters
            al_gen_model.adv_saver.save(sess, '%s/checkpoint' % adv_config.gen_train_dir, global_step=al_gen_model.global_step)
            al_disc_model.adv_saver.save(sess, '%s/checkpoint' % adv_config.disc_train_dir, global_step=al_disc_model.global_step)
            print 'total_batch: ', batch_id, 'gen loss: ', gen_loss, 'disc loss: ', disc_loss
            disc_loss, gen_loss = 0.0, 0.0

def main():
    # Step1: Load data
    gen_data_list, gen_vocab_list, gen_embed, vectors = prepare_origin_data(gen_config)
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True

    # Step2: Train the model
    al_train(gpu_config, gen_config, disc_config, gen_vocab_list, gen_embed, gen_data_list, vectors)

if __name__ == '__main__':
    main()
