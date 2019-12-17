"""Seq2seq loss operations for use in sequence models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
import tensorflow as tf

__all__ = ["sequence_loss"]

def sequence_loss(logits, targets, weights,
                  average_across_timesteps=True, average_across_batch=True,
                  softmax_loss_function=None, name=None):
  """Weighted cross-entropy loss for a sequence of logits (per example).

  Args:
    logits: A 3D Tensor of shape
      [batch_size x sequence_length x num_decoder_symbols] and dtype float.
      The logits correspond to the prediction across all classes at each
      timestep.
    targets: A 2D Tensor of shape [batch_size x sequence_length] and dtype
      int. The target represents the true class at each timestep.
    weights: A 2D Tensor of shape [batch_size x sequence_length] and dtype
      float. Weights constitutes the weighting of each prediction in the
      sequence. When using weights as masking set all valid timesteps to 1 and
      all padded timesteps to 0.
    average_across_timesteps: If set, sum the cost across the sequence
      dimension and divide by the cost by the total label weight across
      timesteps.
    average_across_batch: If set, sum the cost across the batch dimension and
      divide the returned cost by the batch size.
    softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    name: Optional name for this operation, defaults to "sequence_loss".

  Returns:
    A scalar float Tensor: The average log-perplexity per symbol (weighted).

  Raises:
    ValueError: logits does not have 3 dimensions or targets does not have 2
                dimensions or weights does not have 2 dimensions.
  """
  if len(logits.get_shape()) != 3:
    raise ValueError("Logits must be a "
                     "[batch_size x sequence_length x logits] tensor")
  if len(targets.get_shape()) != 2:
    raise ValueError("Targets must be a [batch_size x sequence_length] "
                     "tensor")
  if len(weights.get_shape()) != 2:
    raise ValueError("Weights must be a [batch_size x sequence_length] "
                     "tensor")
  with ops.name_scope(name, "sequence_loss", [logits, targets, weights]):
    num_classes = array_ops.shape(logits)[2]
    max_time = array_ops.shape(logits)[1]
    batch_size = array_ops.shape(logits)[0]
    probs_flat = array_ops.reshape(logits, [-1, num_classes])
    targets = array_ops.reshape(targets, [-1])
    if softmax_loss_function is None:
      crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
        labels=targets, logits=probs_flat)
    else:
      crossent, all_logits = softmax_loss_function(probs_flat, targets)
    crossent = array_ops.reshape(crossent, [-1, max_time]) # batch * len
    all_logits = array_ops.reshape(all_logits, [batch_size, max_time, -1])
    crossent = crossent * weights
    if average_across_timesteps and average_across_batch:
      crossent = math_ops.reduce_sum(crossent, 1)
      total_size = math_ops.reduce_sum(weights, 1)
      total_size += 1e-12 # to avoid division by 0 for all-0 weights
      crossent /= total_size
    else:
      batch_size = array_ops.shape(logits)[0]
      sequence_length = array_ops.shape(logits)[1]
      crossent = array_ops.reshape(crossent, [batch_size, sequence_length])
    if average_across_timesteps and not average_across_batch:
      crossent = math_ops.reduce_sum(crossent, axis=[1])
      total_size = math_ops.reduce_sum(weights, axis=[1])
      total_size += 1e-12 # to avoid division by 0 for all-0 weights
      crossent /= total_size
    if not average_across_timesteps and average_across_batch:
      crossent = math_ops.reduce_sum(crossent, axis=[0])
      total_size = math_ops.reduce_sum(weights, axis=[0])
      total_size += 1e-12 # to avoid division by 0 for all-0 weights
      crossent /= total_size
    return crossent, all_logits
