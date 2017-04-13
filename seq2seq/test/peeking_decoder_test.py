# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Test Cases for decoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
import numpy as np

from seq2seq.decoders import PeekingDecoder

from seq2seq.test.decoder_test import DecoderTests

class PeekingDecoderTest(tf.test.TestCase, DecoderTests):
  """Tests the `PeekingDecoder` class.
  """

  def setUp(self):
    tf.test.TestCase.setUp(self)
    tf.logging.set_verbosity(tf.logging.INFO)
    DecoderTests.__init__(self)
    self.attention_dim = 64
    self.input_seq_len = 10

  def create_decoder(self, helper, mode):
    attention_fn = None
    attention_values = tf.convert_to_tensor(
        np.random.randn(self.batch_size, self.input_seq_len, 32),
        dtype=tf.float32)
    attention_keys = tf.convert_to_tensor(
        np.random.randn(self.batch_size, self.input_seq_len, 32),
        dtype=tf.float32)
    params = PeekingDecoder.default_params()
    params["max_decode_length"] = self.max_decode_length
    return PeekingDecoder(
        params=params,
        mode=mode,
        vocab_size=self.vocab_size,
        attention_keys=attention_keys,
        attention_values=attention_values,
        attention_values_length=np.arange(self.batch_size) + 1,
        attention_fn=attention_fn)

  def test_attention_scores(self):
    decoder_output_ = self.test_with_fixed_inputs()
    np.testing.assert_array_equal(
        decoder_output_.attention_scores.shape,
        [self.sequence_length, self.batch_size, self.input_seq_len])

    # Make sure the attention scores sum to 1 for each step
    scores_sum = np.sum(decoder_output_.attention_scores, axis=2)
    np.testing.assert_array_almost_equal(
        scores_sum, np.ones([self.sequence_length, self.batch_size]))


if __name__ == "__main__":
  tf.test.main()
