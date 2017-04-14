"""
A decoder that peeks at some of the next words in order to choose attention.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
from seq2seq.decoders.attention_decoder import AttentionDecoder, AttentionDecoderOutput

from keras.layers.core import Activation, Dense, RepeatVector
from keras.layers.merge import concatenate
from keras.layers.wrappers import TimeDistributed

class PeekingDecoder(AttentionDecoder):
  """An RNN Decoder that uses attention over an input sequence (and peeks)."""

  def __init__(self, *args, **vargs):
    super().__init__(*args, **vargs)

  @staticmethod
  def one_per_batch(params, indices):
    """Return one slice of params per batch, selected by ints in indices.

    params: batched values
      A tensor of shape [B, ...]
    indices: an index into the 2nd dimension of params
      A tensor of shape [B]
    """
    batch_size = indices.shape[0].value
    batched_indices = tf.transpose(tf.stack([tf.range(batch_size, dtype=indices.dtype), indices]))
    return tf.gather_nd(params, batched_indices)

  def compute_vocab_probs_for_attentions(self, cell_output):
    """Computes the decoder output probabilities under each possible attention."""

    # Generate softmax inputs for each possible attention
    cell_outputs = RepeatVector(self.max_sentence_length)(cell_output)
    merged = concatenate([cell_outputs, self.attention_values])
    softmax_inputs = TimeDistributed(self.softmax_input_layer)(merged)

    # Softmax computation for each possible attention
    return Activation('softmax')(TimeDistributed(self.logit_layer)(softmax_inputs))

  def choose_attentions(self, vocab_probs, correct_words):
    """Compute the best attention index.

    vocab_probs: The vocabulary probabilities under each possible attention
      A tensor of shape [B, T, vocab]
    correct_words: Indices of the correct vocabulary choices
      A tensor of shape [B]
    """
    correct_probs = self.one_per_batch(tf.transpose(vocab_probs, [0, 2, 1]), correct_words)
    return tf.argmax(correct_probs, axis=-1)

  def compute_output(self, cell_output, attentions):
    """Computes the decoder outputs."""
    att_scores = tf.one_hot(attentions, self.attention_values_dim)
    batch_size = cell_output.shape[0].value
    attention_context = self.one_per_batch(self.attention_values, attentions)
    softmax_input = self.softmax_input_layer(concatenate([cell_output, attention_context]))
    logits = self.logit_layer(softmax_input)

    return softmax_input, logits, att_scores, attention_context

  def step(self, time_, inputs, state, name=None):
    cell_output, cell_state = self.cell(inputs, state)

    # setup
    self.max_sentence_length = self.attention_values.shape[-2].value
    self.attention_values_dim = self.attention_values.shape[-1].value
    merged_dim = self.attention_values_dim + cell_output.shape[-1].value
    self.softmax_input_layer = Dense(self.cell.output_size, activation='tanh', input_shape=(merged_dim,))
    self.logit_layer = Dense(self.vocab_size, input_shape=(self.cell.output_size,))

    # find best attentions
    vocab_probs = self.compute_vocab_probs_for_attentions(cell_output)

    # TODO Replace with indices of actual correct words, encoded as one vocabulary index per position
    batch_size = cell_output.shape.dims[0].value
    correct_words = tf.ones(shape=(batch_size,), dtype=tf.int32)

    # compute output
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
