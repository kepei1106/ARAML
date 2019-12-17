import numpy as np
import tensorflow as tf
import sys
import time
import random
import os

sys.path.append("..")

from lm_model import lm_model

random.seed(time.time())

# Evaluation on development set
def evaluate(model, sess, data_dev, batch_size, config):
    loss, acc = np.zeros((1, )), np.zeros((1, ))
    st, ed, times = 0, batch_size, 0
    while st < len(data_dev):
        selected_data = data_dev[st:ed]
        batched_data = model.gen_train_batched_data(selected_data, config)
        outputs = model.step(sess, batched_data, forward_only=True)
        loss += outputs[0]
        st, ed = ed, ed+batch_size
        times += 1
    loss /= times
    return loss

# Infer the sampling probability based on language model
def inference(model, sess, data_test, batch_size, name_model):
    responses = []
    st, ed = 0, batch_size
    while st < len(data_test):
        selected_data = data_test[st:ed]
        batched_data = model.gen_train_batched_data(selected_data, name_model)
        outputs = model.inference(sess, batched_data)[0]
        for idx in range(len(selected_data)):
            responses.append(outputs[idx][batched_data['len_ans'][idx] - 1])
        st, ed = ed, ed+batch_size
    return responses

# Create language model
def create_model(sess, vocab_size, embed_units, units, sequence_length, start_token, end_token, train_dir, name_model, inference_version = 0):
    with tf.variable_scope(name_model):
        model = lm_model(vocab_size, embed_units, units, name_model, sequence_length, start_token, end_token)
        print("Parameters of Language Model:")
        model.print_parameters()

        if tf.train.get_checkpoint_state(train_dir): # Load parameters from existing checkpoint
            if inference_version == 0:
                print("Reading model parameters from %s" % train_dir)
                model.saver.restore(sess, tf.train.latest_checkpoint(train_dir))
            else:
                model_path = '%s/checkpoint-%08d' % (train_dir, inference_version)
                print('Reading model parameters from %s' % model_path)
                model.saver.restore(sess, model_path)
        else:
            print("Created model with fresh parameters.")
            disc_variable = [gv for gv in tf.global_variables() if name_model in gv.name]
            sess.run(tf.variables_initializer(disc_variable))

    return model


# Pre-train the forward/backward language model
def lm_train(model, sess, name_model, pretrain_epoch, per_checkpoint, train_dir, gen_data_loader):
    print('start pre-training %s' % name_model)
    loss_step, time_step = np.zeros((1,)), .0
    previous_losses = [1e18] * 4

    num_batches = gen_data_loader.num_batch
    current_epoch = model.global_step.eval(session = sess) // num_batches
    print 'num_batches = ', num_batches
    print 'current_epoch = ', current_epoch

    for epoch in range(current_epoch, pretrain_epoch):
        for batch in range(num_batches):
            start_time = time.time()
            selected_data = gen_data_loader.next_batch()
            batched_data = model.gen_train_batched_data(selected_data, name_model)
            outputs = model.step(sess, batched_data)
            loss_step += outputs[0] / (per_checkpoint * num_batches)
            time_step += (time.time() - start_time) / (per_checkpoint * num_batches)
        if epoch % per_checkpoint == 0 or epoch == pretrain_epoch - 1:
            show = lambda a: '[%s]' % (' '.join(['%.2f' % x for x in a]))
            print("epoch %d learning rate %.4f step-time %.2f loss %s perplexity %s"
                  % (epoch, model.learning_rate.eval(session = sess),
                     time_step, show(loss_step), show(np.exp(loss_step))))
            model.saver.save(sess, '%s/checkpoint' % train_dir, global_step=model.global_step)
            if np.sum(loss_step) > max(previous_losses):
                sess.run(model.learning_rate_decay_op)
            previous_losses = previous_losses[1:] + [np.sum(loss_step)]
            loss_step, time_step = np.zeros((1,)), .0


# Output the probability based on the language model
def lm_test(model, sess, data_test, batch_size, name_model):
    responses = inference(model, sess, data_test, batch_size, name_model)
    return responses
