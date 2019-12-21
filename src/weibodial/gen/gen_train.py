import numpy as np
import tensorflow as tf
import sys
import time
import random
import os

sys.path.append("..")

import utils.conf as conf
import utils.data_utils as data_utils
from gen_model import gen_model

SEED = int(time.time())
random.seed(SEED)
np.random.seed(SEED)

# Generate responses during inference
def inference(model, sess, data_test, batch_size):
    # Get the generated results from the generator
    responses = []
    st, ed = 0, batch_size
    while st < len(data_test):
        selected_data = data_test[st:ed]
        batched_data = model.gen_train_batched_data(selected_data)
        outputs = model.inference(sess, batched_data)
        responses.extend(outputs[0])
        st, ed = ed, ed+batch_size

    # Keep the text before _EOS
    results = []
    for response in responses:
        result = []
        for token in response:
            if token != '_EOS':
                result.append(token)
            else:
                break
        results.append(result)

    return results


# Create a seq2seq model for the generator
def create_model(sess, model_config, vocab, embed, train_dir, inference_version = 0):
    with tf.variable_scope(model_config.name_model):
        model = gen_model(model_config.vocab_size, model_config.embed_units, model_config.units, model_config.num_layers, vocab=vocab, embed=embed,
                          learning_rate=model_config.lr, learning_rate_decay_factor=model_config.lr_decay, name_scope = model_config.name_model,
                          max_gradient_norm=model_config.max_gradient_norm)
        if model_config.log_parameters:
            print("Parameters of %s:" % model_config.name_model)
            model.print_parameters()

        if tf.train.get_checkpoint_state(train_dir):
            if inference_version == 0:
                print("Reading model parameters from %s" % train_dir)
                model.saver.restore(sess, tf.train.latest_checkpoint(train_dir))
                print 'latest'
            else:
                model_path = '%s/checkpoint-%08d' % (train_dir, inference_version)
                print('Reading model parameters from %s' % model_path)
                model.saver.restore(sess, model_path)
                print 'appointed'
        else:
            print("Created model with fresh parameters.")
            disc_variable = [gv for gv in tf.global_variables() if model_config.name_model in gv.name]
            sess.run(tf.variables_initializer(disc_variable))

        model.symbol2index.init.run()
    return model

# Pre-train the generator
def gen_pretrain(model, sess, model_config, data_train):
    print('start pre-training generator')
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
            print("epoch %d learning rate %.4f step-time %.2f loss %s perplexity %s"
                  % (epoch, model.learning_rate.eval(session = sess),
                     time_step, show(loss_step), show(np.exp(loss_step))))
            model.saver.save(sess, '%s/checkpoint' % model_config.train_dir, global_step=model.global_step)
            if np.sum(loss_step) > max(previous_losses):
                sess.run(model.learning_rate_decay_op)
            previous_losses = previous_losses[1:] + [np.sum(loss_step)]
            loss_step, time_step = np.zeros((1,)), .0

        np.random.shuffle(data_train)

# Generate negative samples for the discriminator
def generate_negative_samples(sess, model, sample_num, data, output_dir, batch_size):
    # Positive samples: Negative samples=1:1
    if sample_num < len(data):
        np.random.shuffle(data)
        sample_data = data[:sample_num]
    else:
        sample_data = data

    neg_samples = inference(model, sess, sample_data, batch_size)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open('%s/post_samples.txt' % (output_dir) , 'w') as f:
        for i in range(len(sample_data)):
            f.writelines('%s\n' % (' '.join(sample_data[i]['query'])))

    with open('%s/positive_samples.txt' % (output_dir), 'w') as f:
        for i in range(len(sample_data)):
            f.writelines('%s\n' % (' '.join(sample_data[i]['ans'])))

    with open('%s/negative_samples.txt' % (output_dir), 'w') as f:
        for i in range(len(neg_samples)):
            f.writelines('%s\n' % (' '.join(neg_samples[i])))

# Acquire the generated results
def gen_test(sess, model, data, output_dir, batch_size, batch_id):
    results = inference(model, sess, data, batch_size)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open('%s/generation_%d.txt' % (output_dir, batch_id) , 'w') as f:
        for i in range(len(data)):
            f.writelines('%s\n' % (' '.join(data[i]['query'])))
            f.writelines('%s\n' % (' '.join(data[i]['ans'])))
            f.writelines('%s\n' % (' '.join(results[i])))
            f.writelines('\n')
