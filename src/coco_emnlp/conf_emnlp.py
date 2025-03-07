import os

class adv_config(object):
    # Pre-training / Adversarial Training Hyper-Parameters
    ADV_LOG_STEP = 5
    tau = 0.9  # temperature for sampling
    log_dir = 'log_emnlp'
    result_dir = 'res_emnlp'
    train_dir = 'train_emnlp'
    TOTAL_BATCH = 600 # Training batches in adversarial training
    data_dir = '../../data/emnlp'
    positive_file = os.path.join(data_dir, 'emnlp_train.txt') # Training set directory
    negative_file = os.path.join(log_dir, 'generator_sample' + str(tau) + '.txt')
    vocab_file = os.path.join(data_dir, 'vocab_emnlp.txt') # Vocabulary directory
    eval_file_prefix = os.path.join(log_dir, 'evaler_file' + str(tau))
    result_prefix = os.path.join(result_dir, 'result_file' + str(tau))
    raml_file = 'raml_emnlp.txt'
    generated_num = 2000
    result_generated_num = 20000
    generated_num_pre = 20000
    restore = False  # whether to restore the checkpoint
    vocab_size = 5722  # vocab size of coco
    sample_size = 5  # the number of samples from Ps
    gen_update_batch = 500
    dis_update_batch = 2
    adv_generate_num = 2
    alpha = 0.2  # rescaling coefficient
    PAD_ID = 5721 # Token ID of PAD

class disc_config(object):
    # Discriminator Hyper-Parameters
    MID_LAYER_G = [256]
    MID_LAYER_R = [512]
    re_dropout_keep_prob = 0.75  # keep probability in discriminator
    re_l2_reg_lambda = 1e-1  # regularization coefficient
    R_rate = 0.0001  # learning rate of discriminator
    PRE_GENERATE_NUM = 1 # pretraining epochs = PRE_GENERATE_NUM * PRE_EPOCH_NUM_D
    PRE_EPOCH_NUM_D = 15

class gen_config(object):
    # Generator Hyper-Parameters
    EMB_DIM = 128  # embedding dimension
    HIDDEN_DIM = 128  # hidden state dimension of lstm/gru cell
    SEQ_LENGTH = 49  # sequence length
    START_TOKEN = 0
    PRE_EPOCH_NUM = 50  # MLE pretraining epochs
    PRE_LOG_STEP = 5
    BATCH_SIZE = 100
    R_rate_gen = 0.001  # learning rate of generator

class lm_config(object):
    # Language Model Hyper-Parameters
    lm_fw_traindir = './lm_train_fw_emnlp'
    lm_bw_traindir = './lm_train_bw_emnlp'
