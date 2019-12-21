import numpy as np
import tensorflow as tf
import sys
import time
import random
import os

sys.path.append("..")

import utils.conf as conf
import utils.data_utils as data_utils
from disc_model import disc_model

SEED = int(time.time())
random.seed(SEED)
np.random.seed(SEED)

# Assign rewards to samples
def inference(model, sess, data_test, batch_size):
    score = []
    st, ed = 0, batch_size
    while st < len(data_test):
        selected_data = data_test[st:ed]
        batched_data = model.gen_test_batched_data(selected_data)
        outputs = model.inference(sess, batched_data)
        score += outputs[0]
        st, ed = ed, ed+batch_size
    return score

# Create model for discriminator
def create_model(sess, model_config, vocab, embed, inference_version = 0):
    with tf.variable_scope(model_config.name_model):
        model = disc_model(model_config.vocab_size, model_config.embed_units, model_config.units, vocab=vocab, embed=embed,
                           learning_rate=model_config.lr, learning_rate_decay_factor=model_config.lr_decay, name_scope = model_config.name_model,
                           max_gradient_norm=model_config.max_grad_norm, l2_lambda=model_config.l2_lambda)

        if model_config.log_parameters:
            print("Parameters of %s:" % model_config.name_model)
            model.print_parameters()

        if tf.train.get_checkpoint_state(model_config.train_dir):
            if inference_version == 0:
                print("Reading model parameters from %s" % model_config.train_dir)
                model.saver.restore(sess, tf.train.latest_checkpoint(model_config.train_dir))
            else:
                model_path = '%s/checkpoint-%08d' % (model_config.train_dir, inference_version)
                print('Reading model parameters from %s' % model_path)
                model.saver.restore(sess, model_path)
        else:
            print("Created model with fresh parameters.")
            disc_variable = [gv for gv in tf.global_variables() if model_config.name_model in gv.name]
            sess.run(tf.variables_initializer(disc_variable))

        model.symbol2index.init.run()
    return model

# Train the discriminator
def disc_train(model, sess, model_config, data_train):
    print('start pre-training discriminator')
    loss_step, time_step = np.zeros((1,)), .0
    previous_losses = [1e18] * 4

    num_batches = len(data_train) // model_config.batch_size
    current_epoch = model.global_step.eval(session = sess) // num_batches
    print 'num_batches = ', num_batches
    print 'current_epoch = ', current_epoch

    for epoch in range(current_epoch, model_config.pretrain_epoch):
        for batch in range(num_batches):
            start_time = time.time()
            selected_data = data_train[batch*model_config.batch_size: (batch+1)*model_config.batch_size]
            batched_data = model.gen_train_batched_data(selected_data)
            outputs = model.step(sess, batched_data)
            loss_step += outputs[0] / (model_config.per_checkpoint * num_batches)
            time_step += (time.time() - start_time) / (model_config.per_checkpoint * num_batches)
        if epoch % model_config.per_checkpoint == 0:
            show = lambda a: '[%s]' % (' '.join(['%.2f' % x for x in a]))
            print("epoch %d learning rate %.4f step-time %.2f loss %s"
                  % (epoch, model.learning_rate.eval(session = sess),
                     time_step, show(loss_step)))
            model.saver.save(sess, '%s/checkpoint' % model_config.train_dir, global_step=model.global_step)

            if np.sum(loss_step) > max(previous_losses):
                sess.run(model.learning_rate_decay_op)
            previous_losses = previous_losses[1:] + [np.sum(loss_step)]
            loss_step, time_step = np.zeros((1,)), .0

        np.random.shuffle(data_train)

# Acquire the reward during inference
def disc_test(gpu_config, data_test, batch_size):
    with tf.Session(config=gpu_config) as sess:
        model = create_model(sess, model_config, None, None, 0)
    return inference(model, sess, data_test, batch_size)

