import numpy as np
import tensorflow as tf
import random
import os
import math
import argparse
from araml_dataloader import Gen_Data_loader, Dis_dataloader
from araml_generator import Generator
from araml_rewarder import Rewarder
from convert import convert

import os
import time
import lm.lm_train as lm_train

SEED = int(time.time()) # random seed

parser = argparse.ArgumentParser()

parser.add_argument('--task_name', default='coco', type=str, required=True,
                    help='The name of task, i.e. coco or emnlp')

args = parser.parse_args()

if args.task_name == 'coco':
    from conf_coco import gen_config, disc_config, adv_config, lm_config
    print('coco')
else:
    from conf_emnlp import gen_config, disc_config, adv_config, lm_config
    print('emnlp')

#  Generator Hyper-Parameters Initialization
EMB_DIM = gen_config.EMB_DIM # embedding dimension
HIDDEN_DIM = gen_config.HIDDEN_DIM # hidden state dimension of lstm cell
SEQ_LENGTH = gen_config.SEQ_LENGTH # sequence length
START_TOKEN = gen_config.START_TOKEN
PRE_EPOCH_NUM = gen_config.PRE_EPOCH_NUM # MLE pretraining epochs
PRE_LOG_STEP = gen_config.PRE_LOG_STEP
BATCH_SIZE = gen_config.BATCH_SIZE
R_rate_gen = gen_config.R_rate_gen # learning rate of generator

#  Discriminator Hyper-Parameters Initialization
MID_LAYER_G = disc_config.MID_LAYER_G
MID_LAYER_R = disc_config.MID_LAYER_R
re_dropout_keep_prob = disc_config.re_dropout_keep_prob # keep probability in discriminator
re_l2_reg_lambda = disc_config.re_l2_reg_lambda # regularization coefficient
R_rate = disc_config.R_rate # learning rate of discriminator
PRE_GENERATE_NUM = disc_config.PRE_GENERATE_NUM # pretraining epochs = PRE_GENERATE_NUM * PRE_EPOCH_NUM_D
PRE_EPOCH_NUM_D = disc_config.PRE_EPOCH_NUM_D

#  Language Model Hyper-Parameters Initialization
lm_fw_traindir = lm_config.lm_fw_traindir
lm_bw_traindir = lm_config.lm_bw_traindir

#  Pre-training / Adversarial Training Hyper-Parameters
ADV_LOG_STEP = adv_config.ADV_LOG_STEP
tau = adv_config.tau # temperature for sampling
log_dir = adv_config.log_dir
result_dir = adv_config.result_dir
train_dir = adv_config.train_dir
TOTAL_BATCH = adv_config.TOTAL_BATCH # Training batches in adversarial training
positive_file = adv_config.positive_file # Training set directory
negative_file = adv_config.negative_file
vocab_file = adv_config.vocab_file # Vocabulary directory
eval_file_prefix = adv_config.eval_file_prefix
result_prefix = adv_config.result_prefix
raml_file = adv_config.raml_file
generated_num = adv_config.generated_num
result_generated_num = adv_config.result_generated_num
generated_num_pre = adv_config.generated_num_pre
restore = adv_config.restore # whether to restore the checkpoint
vocab_size = adv_config.vocab_size # vocab size of coco
sample_size = adv_config.sample_size # the number of samples from Ps
gen_update_batch = adv_config.gen_update_batch
dis_update_batch = adv_config.dis_update_batch
adv_generate_num = adv_config.adv_generate_num
alpha = adv_config.alpha # rescaling coefficient
PAD_ID = adv_config.PAD_ID # Token ID of PAD


# Generate samples to file
def generate_samples(sess, trainable_model, batch_size, generated_num, output_file):
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        samples = trainable_model.generate(sess)
        generated_samples.extend(samples)

    with open(output_file, 'w') as fout:
        for sample in generated_samples:
            buffer = ' '.join([str(x) for x in sample]) + '\n'
            fout.write(buffer)

    return generated_samples


# Training process (one epoch)
def pre_train_epoch(sess, trainable_model, data_loader):
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss = trainable_model.pretrain_step(sess, batch, np.ones(BATCH_SIZE*sample_size))
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


def main():
    # Set random seed
    random.seed(SEED)
    np.random.seed(SEED)

    # Construct generator / discriminator dataloader
    gen_data_loader = Gen_Data_loader(BATCH_SIZE, tau, vocab_size, sample_size)
    dis_data_loader = Dis_dataloader(BATCH_SIZE)
    gen_data_loader.create_batches(positive_file, PAD_ID, SEQ_LENGTH)

    # Create generator / discriminator model
    generator = Generator(vocab_size, BATCH_SIZE * sample_size, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN, MID_LAYER_G, R_rate_gen)
    rewarder = Rewarder(vocab_size, BATCH_SIZE, EMB_DIM * 2, HIDDEN_DIM * 2, SEQ_LENGTH, l2_reg_lambda=re_l2_reg_lambda)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # Saver for pretraining and adversarial training
    saver_pre = tf.train.Saver(tf.global_variables(), max_to_keep=5, keep_checkpoint_every_n_hours=1.0)
    saver_adv = tf.train.Saver(tf.global_variables(), max_to_keep=5, keep_checkpoint_every_n_hours=1.0)

    # Construct the forward/backward language model for sampling
    lm_model_fw = lm_train.create_model(sess, vocab_size, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN, PAD_ID, lm_fw_traindir, 'lm_model_fw')
    lm_model_bw = lm_train.create_model(sess, vocab_size, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN, PAD_ID, lm_bw_traindir, 'lm_model_bw')
    lm_train.lm_train(lm_model_fw, sess, 'lm_model_fw', PRE_EPOCH_NUM, PRE_LOG_STEP, lm_fw_traindir , gen_data_loader)
    lm_train.lm_train(lm_model_bw, sess, 'lm_model_bw', PRE_EPOCH_NUM, PRE_LOG_STEP, lm_bw_traindir, gen_data_loader)
    gen_data_loader.create_batches_augment(raml_file, lm_model_fw, lm_model_bw, sess, PAD_ID)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log = open(os.path.join(log_dir, 'experiment-log-'+str(tau)+'.txt'), 'w')

    if restore is False:
        print 'Start pre-training generator...'
        log.write('pre-training...\n')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        for epoch in xrange(PRE_EPOCH_NUM):
            loss = pre_train_epoch(sess, generator, gen_data_loader)
            if epoch % PRE_LOG_STEP == 0:
                print 'pre-train epoch ', epoch, 'test_loss ', loss
                buffer = 'epoch:\t' + str(epoch) + '\tnll:\t' + str(loss) + '\n'
                log.write(buffer)
                generate_samples(sess, generator, BATCH_SIZE * sample_size, result_generated_num,
                                 result_prefix + '_pre_' + str(epoch))

        print 'Start pre-training discriminator...'
        start = time.time()
        for _ in range(PRE_GENERATE_NUM):
            generate_samples(sess, generator, BATCH_SIZE, generated_num_pre, negative_file)
            dis_data_loader.load_train_data(positive_file, negative_file, SEQ_LENGTH)

            for _ in range(PRE_EPOCH_NUM_D):
                dis_data_loader.reset_pointer()
                r_losses = []
                for it in xrange(dis_data_loader.num_batch):
                    x_text, y_label = dis_data_loader.next_batch()
                    _, r_loss = rewarder.reward_train_step(sess, x_text, y_label, R_rate, re_dropout_keep_prob)
                    r_losses.append(r_loss)
                print 'discriminator_loss', np.mean(r_losses)
        speed = time.time() - start
        print 'Discriminator pre_training Speed:{:.3f}'.format(speed)

        train_dir_pre = os.path.join(train_dir, 'pre')
        if not os.path.exists(train_dir_pre):
            os.makedirs(train_dir_pre)

        checkpoint_path = os.path.join(train_dir_pre, 'exper_pre.ckpt')
        saver_pre.save(sess, checkpoint_path)
    else:
        print 'Restore pretrained model ...'
        log.write('Restore pre-trained model...\n')
        train_dir_pre = os.path.join(train_dir, 'pre')
        ckpt = tf.train.get_checkpoint_state(train_dir_pre)
        saver_pre.restore(sess, ckpt.model_checkpoint_path)

    print 'Start Adversarial Training...'
    log.write('adversarial training...\n')

    train_dir_adv = os.path.join(train_dir, 'adv')
    if not os.path.exists(train_dir_adv):
        os.makedirs(train_dir_adv)

    for total_batch in range(TOTAL_BATCH):
        if total_batch % ADV_LOG_STEP == 0 or total_batch == TOTAL_BATCH - 1:
            generate_samples(sess, generator, BATCH_SIZE * sample_size, generated_num, eval_file_prefix + '_batch_'+ str(total_batch))
            generate_samples(sess, generator, BATCH_SIZE * sample_size, result_generated_num, result_prefix + '_batch_' + str(total_batch))
            checkpoint_path = 'exper_' + str(total_batch) + '.ckpt'
            checkpoint_path = os.path.join(train_dir_adv, checkpoint_path)
            saver_adv.save(sess, checkpoint_path)

            # convert token id to token
            convert(eval_file_prefix + '_batch_'+ str(total_batch), vocab_file, log_dir)

        # Train the generator
        start = time.time()
        g_losses = []
        for it in range(gen_update_batch):
            batch_x = gen_data_loader.next_batch_augment() # (batch_size * sample_size, sequence_length)
            rewards = rewarder.get_reward(sess, batch_x)[0]
            weights = []
            for it3 in range(BATCH_SIZE):
                tmp_weights = rewards[it3*sample_size:(it3+1)*sample_size]
                exp_weights = [math.exp(tmp_weights[w] / alpha) for w in range(sample_size)]
                weights.extend(exp_weights)
            normalizer = sum(weights)
            weights = np.array([w / normalizer for w in weights])
            _, g_loss = generator.pretrain_step(sess, batch_x, weights)
            g_losses.append(g_loss)
        speed = time.time() - start
        print 'Generator training {} round, Speed:{:.3f}, Loss:{:.3f}'.format(total_batch, speed, np.mean(g_losses))

        # Train the discriminator
        start = time.time()
        r_loss_list = []
        for _ in range(adv_generate_num):
            generate_samples(sess, generator, BATCH_SIZE * sample_size, generated_num, negative_file)
            dis_data_loader.load_train_data(positive_file, negative_file, SEQ_LENGTH)
            for _ in range(dis_update_batch):
                dis_data_loader.reset_pointer()
                for it in xrange(dis_data_loader.num_batch):
                    x_text, y_label = dis_data_loader.next_batch()
                    _, r_loss = rewarder.reward_train_step(sess, x_text, y_label, R_rate, re_dropout_keep_prob)
                    r_loss_list.append(r_loss)
        avg_loss = np.mean(r_loss_list)
        speed = time.time() - start
        print 'Discriminator training {} round, Speed:{:.3f}, Loss:{:.3f}'.format(total_batch, speed, avg_loss)

    log.close()

if __name__ == '__main__':
    main()
