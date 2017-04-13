"""
A decoder that peeks at some of the next words in order to choose attention.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
import numpy as np
from seq2seq.decoders.attention_decoder import AttentionDecoder, AttentionDecoderOutput

from seq2seq.contrib.seq2seq.helper import CustomHelper

from tensorflow.python.framework import function  # pylint: disable=E0611

from keras.layers.core import Activation, Dense, RepeatVector
from keras.layers.merge import Concatenate
from keras.layers.wrappers import TimeDistributed

class PeekingDecoder(AttentionDecoder):
  """An RNN Decoder that uses attention over an input sequence.

  Args:
    cell: An instance of ` tf.contrib.rnn.RNNCell`
    helper: An instance of `tf.contrib.seq2seq.Helper` to assist decoding
    initial_state: A tensor or tuple of tensors used as the initial cell
      state.
    vocab_size: Output vocabulary size, i.e. number of units
      in the softmax layer
    attention_keys: The sequence used to calculate attention scores.
      A tensor of shape `[B, T, ...]`.
    attention_values: The sequence to attend over.
      A tensor of shape `[B, T, input_dim]`.
    attention_values_length: Sequence length of the attention values.
      An int32 Tensor of shape `[B]`.
    attention_fn: The attention function to use. This function map from
      `(state, inputs)` to `(attention_scores, attention_context)`.
      For an example, see `seq2seq.decoder.attention.AttentionLayer`.
    reverse_scores: Optional, an array of sequence length. If set,
      reverse the attention scores in the output. This is used for when
      a reversed source sequence is fed as an input but you want to
      return the scores in non-reversed order.
  """

  def __init__(self, *args, **vargs):
    super().__init__(*args, **vargs)
    self.attention_values_dim = self.attention_values.shape.dims[-1].value
    self.softmax_input_layer = Dense(self.cell.output_size, activation='tanh')
    self.logit_layer = Dense(self.vocab_size)

  def compute_vocab_probs_for_attentions(self, cell_output):
    """Computes the decoder output probabilities under each possible attention."""

    # Generate softmax inputs for each possible attention
    cell_outputs = RepeatVector(self.attention_values_length)(cell_output)
    softmax_inputs = TimeDistributed(self.softmax_input_layer)(
      Concatenate([cell_outputs, self.attention_values], axis=-1))

    # Softmax computation for each possible attention
    return TimeDistributed(Activation('softmax')(self.logit_layer))(softmax_inputs)

  def choose_attentions(self, vocab_probs, correct_words):
    """Compute the best attention index.

    vocab_probs: The vocabulary probabilities under each possible attention
      A tensor of shape [B, T, vocab]
    correct_words: Indices of the correct vocabulary choices
      A tensor of shape [B]
    """
    correct_probs = tf.gather_nd(tf.transpose(vocab_probs, [0, 2, 1]), correct_words)
    return tf.argmax(correct_probs, axis=-1)

  def compute_output(self, cell_output, attentions):
    """Computes the decoder outputs."""

    att_scores = tf.one_hot(attentions, self.attention_values_dim)
    attention_context = tf.gather_nd(self.attention_values, attentions)
    softmax_input = self.softmax_input_layer(Concatenate([cell_output, attention_context]))
    logits = self.logit_layer(softmax_input)

    return softmax_input, logits, att_scores, attention_context

  def step(self, time_, inputs, state, name=None):
    cell_output, cell_state = self.cell(inputs, state)
    vocab_probs = self.compute_vocab_probs_for_attentions(cell_output)

    # TODO Replace with actual correct words, encoded as one vocabulary index per position
    correct_words = tf.ones(shape=(4, 16))

    attentions = self.choose_attentions(vocab_probs, correct_words)
    cell_output_new, logits, attention_scores, attention_context = \
      self.compute_output(cell_output, attentions)

    ####################################################
    # Everything below is copied from AttentionDecoder #
    ####################################################

    if self.reverse_scores_lengths is not None:
      attention_scores = tf.reverse_sequence(
          input=attention_scores,
          seq_lengths=self.reverse_scores_lengths,
          seq_dim=1,
          batch_dim=0)

    sample_ids = self.helper.sample(
        time=time_, outputs=logits, state=cell_state)

    outputs = AttentionDecoderOutput(
        logits=logits,
        predicted_ids=sample_ids,
        cell_output=cell_output_new,
        attention_scores=attention_scores,
        attention_context=attention_context)

    finished, next_inputs, next_state = self.helper.next_inputs(
        time=time_, outputs=outputs, state=cell_state, sample_ids=sample_ids)

    return (outputs, next_state, next_inputs, finished)
