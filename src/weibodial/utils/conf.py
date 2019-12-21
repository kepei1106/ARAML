class adv_config(object):
    # Adversarial Training Hyper-Parameters
    total_batch = 1000 # Total batches for adversarial training
    gen_update_time = 1000
    dis_update_time = 2
    dis_epoch_per_update = 1
    gen_train_dir = './gen_train_adv' # Directory to save G's parameters
    disc_train_dir = './disc_train_adv' # Directory to save D's parameters
    test_per_checkpoint = 5
    batch_size = 100


class disc_config(object):
    # Discriminator Hyper-Parameters
    batch_size = 100
    lr = 1e-4 # learning rate
    lr_decay = 0.995 # learning rate decay
    vocab_size = 8002 # size of vocabulary
    log_parameters = True
    embed_units = 100 # dimension of word embedding
    units = 128 # dimension of hidden state
    steps_per_checkpoint = 200
    data_dir = './disc_data' # Directory for the training data of D
    train_dir = './disc_train' # Directory for D's parameters during pretraining
    name_model = "disc_model"
    max_grad_norm = 5 # maximum gradient norm
    l2_lambda = 0.2 # coefficient of regularization
    first_sample_num = 50000
    sample_num = 1000
    pretrain_epoch = 10
    per_checkpoint = 1


class gen_config(object):
    # Generator Hyper-Parameters
    lr = 0.001 # learning rate
    lr_decay = 0.99 # learning rate decay
    max_gradient_norm = 5.0 # maximum gradient norm
    batch_size = 100
    embed_units = 100 # dimension of word embedding
    units = 128 # dimension of hidden state
    num_layers = 2 # the number of RNN layers
    vocab_size = 8002 # size of vocabulary
    log_parameters = True
    data_dir = "../../data/weibodial" # Directory of real data
    raml_data_dir = './raml_data' # Directory of sampled data
    train_dir = "./gen_train" # Directory of G's parameters during pretraining
    vector_dir = "../../data" # Directory of word vector
    name_model = "gen_model"
    test_dir = './gen_test'
    per_checkpoint = 1
    pretrain_epoch = 50

class lm_forward_config(object):
    # Forward Language Model Hyper-Parameters
    learning_rate = 0.001
    learning_rate_decay_factor = 0.95
    max_gradient_norm = 5.0
    batch_size = 100
    embed_units = 100 # dimension of word embedding
    units = 128 # dimension of hidden state
    vocab_size = 8002 # size of vocabulary
    log_parameters = True
    train_dir = "./lm_train_fw"
    name_model = "lm_model_fw"
    per_checkpoint = 1
    pretrain_epoch = 30
    direction = 1 # 1 indicates forward


class lm_backward_config(object):
    # Backward Language Model Hyper-Parameters
    learning_rate = 0.001
    learning_rate_decay_factor = 0.95
    max_gradient_norm = 5.0
    batch_size = 100
    embed_units = 100 # dimension of word embedding
    units = 128 # dimension of hidden state
    vocab_size = 8002 # size of vocabulary
    log_parameters = True
    train_dir = "./lm_train_bw"
    name_model = "lm_model_bw"
    per_checkpoint = 1
    pretrain_epoch = 30
    direction = 0 # 0 indicates backward
