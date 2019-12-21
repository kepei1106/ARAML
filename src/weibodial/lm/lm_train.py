import numpy as np
import tensorflow as tf
import sys
import time
import os

sys.path.append("..")

import utils.conf as conf
import utils.data_utils as data_utils
from lm_model import lm_model

# Calculate the language model score
def inference(model, sess, data_test, batch_size, config):
    responses = []
    st, ed = 0, batch_size
    while st < len(data_test):
        selected_data = data_test[st:ed]
        batched_data = model.gen_train_batched_data(selected_data, config)
        outputs = model.inference(sess, batched_data)[0]
        for idx in range(len(selected_data)):
            responses.append(outputs[idx][batched_data['len_ans'][idx] - 1])
        st, ed = ed, ed+batch_size

    return responses


# Create language model for sampling
def create_model(sess, model_config, vocab, embed, inference_version = 0):
    with tf.variable_scope(model_config.name_model):
        model = lm_model(model_config.vocab_size, model_config.embed_units, model_config.units,
                vocab=vocab, embed=embed, name_scope = model_config.name_model, learning_rate=model_config.learning_rate,
                learning_rate_decay_factor=model_config.learning_rate_decay_factor, max_gradient_norm=model_config.max_gradient_norm)
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

# Pre-train forward / backward language model
def lm_train(model, sess, model_config, data_list):
    data_train, data_test = data_list[0], data_list[1]

    print('start pre-training %s' % model_config.name_model)
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
            batched_data = model.gen_train_batched_data(selected_data, model_config)
            outputs = model.step(sess, batched_data)
            loss_step += outputs[0] / (model_config.per_checkpoint * num_batches)
            time_step += (time.time() - start_time) / (model_config.per_checkpoint * num_batches)
        if epoch % model_config.per_checkpoint == 0:
            show = lambda a: '[%s]' % (' '.join(['%.2f' % x for x in a]))
            print("epoch %d learning rate %.4f step-time %.2f loss %s perplexity %s"
                  % (epoch, model.learning_rate.eval(session = sess), time_step, show(loss_step), show(np.exp(loss_step))))
            model.saver.save(sess, '%s/checkpoint' % model_config.train_dir, global_step=model.global_step)
            if np.sum(loss_step) > max(previous_losses):
                sess.run(model.learning_rate_decay_op)
            previous_losses = previous_losses[1:] + [np.sum(loss_step)]
            loss_step, time_step = np.zeros((1,)), .0

# Get the language model score
def lm_test(model, sess, data_test, batch_size, config):
    responses = inference(model, sess, data_test, batch_size, config)
    return responses
