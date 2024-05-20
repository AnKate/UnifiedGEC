# coding=utf-8
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Beam search to find the translated sequence with the highest probability.

Source implementation from Tensor2Tensor:
https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/beam_search.py
"""

import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils


# from torch.nn.utils.rnn import map_structure


def inf(dtype):
    """Returns a value close to infinity, but is still finite in `dtype`.

    This is useful to get a very large value that is still zero when multiplied by
    zero. The floating-point "Inf" value is NaN when multiplied by zero.

    Args:
        dtype: A dtype. The returned value will be finite when casted to this dtype.

    Returns:
        A very large value.
    """
    if dtype == torch.float32 or dtype == torch.bfloat16:
        return 1e7
    elif dtype == torch.float16:
        return np.finfo(np.float16).max
    else:
        raise AssertionError('Invalid dtype: %s' % dtype)


class _StateKeys(object):
    """Keys to dictionary storing the state of the beam search loop."""

    # Variable storing the loop index.
    CUR_INDEX = "CUR_INDEX"

    # Top sequences that are alive for each batch item. Alive sequences are ones
    # that have not generated an EOS token. Sequences that reach EOS are marked as
    # finished and moved to the FINISHED_SEQ tensor.
    # Has shape [batch_size, beam_size, CUR_INDEX + 1]
    ALIVE_SEQ = "ALIVE_SEQ"
    # Log probabilities of each alive sequence. Shape [batch_size, beam_size]
    ALIVE_LOG_PROBS = "ALIVE_LOG_PROBS"
    # Dictionary of cached values for each alive sequence. The cache stores
    # the encoder output, attention bias, and the decoder attention output from
    # the previous iteration.
    ALIVE_CACHE = "ALIVE_CACHE"

    # Top finished sequences for each batch item.
    # Has shape [batch_size, beam_size, CUR_INDEX + 1]. Sequences that are
    # shorter than CUR_INDEX + 1 are padded with 0s.
    FINISHED_SEQ = "FINISHED_SEQ"
    # Scores for each finished sequence. Score = log probability / length norm
    # Shape [batch_size, beam_size]
    FINISHED_SCORES = "FINISHED_SCORES"
    # Flags indicating which sequences in the finished sequences are finished.
    # At the beginning, all of the sequences in FINISHED_SEQ are filler values.
    # True -> finished sequence, False -> filler. Shape [batch_size, beam_size]
    FINISHED_FLAGS = "FINISHED_FLAGS"


# 为map_structure增加的函数
def expand_to_beam_size(t, beam_size):
    """Expands tensor `t` to have beam size on first dimension."""
    s = t.shape
    return torch.zeros((s[0] * beam_size,) + s[1:], dtype=t.dtype, device=t.device)


# 为map_structure增加的函数
def map_structure(func, *structure):
    """Applies `func` to each element in the given nested structure."""
    return [func(*x) if isinstance(x, tuple) else func(x) for x in zip(*structure)]


class SequenceBeamSearch(object):
    """Implementation of beam search loop."""

    def __init__(self,
                 symbols_to_logits_fn,
                 vocab_size,
                 batch_size,
                 beam_size,
                 alpha,
                 max_decode_length,
                 eos_id,
                 padded_decode,
                 dtype=torch.float32):
        """Initialize sequence beam search.

        Args:
          symbols_to_logits_fn: A function to provide logits, which is the
            interface to the Transformer model. The passed in arguments are:
              ids -> A tensor with shape [batch_size * beam_size, index].
              index -> A scalar.
              cache -> A nested dictionary of tensors [batch_size * beam_size, ...].
            The function must return a tuple of logits and the updated cache:
              logits -> A tensor with shape [batch * beam_size, vocab_size].
              updated cache -> A nested dictionary with the same structure as the
                input cache.
          vocab_size: An integer, the size of the vocabulary, used for topk
            computation.
          batch_size: An integer, the decode batch size.
          beam_size: An integer, number of beams for beam search.
          alpha: A float, defining the strength of length normalization.
          max_decode_length: An integer, the maximum number of steps to decode
            a sequence.
          eos_id: An integer. ID of end of sentence token.
          padded_decode: A bool, indicating if max_sequence_length padding is used
            for beam search.
          dtype: A pytorch data type used for score computation. The default is
            torch.float32.
        """
        self.symbols_to_logits_fn = symbols_to_logits_fn
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.alpha = alpha
        self.max_decode_length = max_decode_length
        self.eos_id = eos_id
        self.padded_decode = padded_decode
        self.dtype = dtype

    def search(self, initial_ids, initial_cache):
        """Beam search for sequences with highest scores."""
        state, state_shapes = self._create_initial_state(initial_ids, initial_cache)

        while self._continue_search(*state):
            state = self._search_step(*state)

        alive_seq = state[0][_StateKeys.ALIVE_SEQ]
        alive_log_probs = state[0][_StateKeys.ALIVE_LOG_PROBS]
        finished_seq = state[0][_StateKeys.FINISHED_SEQ]
        finished_scores = state[0][_StateKeys.FINISHED_SCORES]
        finished_flags = state[0][_StateKeys.FINISHED_FLAGS]

        # Account for corner case where there are no finished sequences for a
        # particular batch item. In that case, return alive sequences for that batch
        # item.
        finished_seq = torch.where(
            torch.any(finished_flags, dim=1, keepdim=True), finished_seq, alive_seq
        )
        finished_scores = torch.where(
            torch.any(finished_flags, dim=1, keepdim=True), finished_scores, alive_log_probs
        )
        return finished_seq, finished_scores

    # def search(self, initial_ids, initial_cache):
    #     """Beam search for sequences with highest scores."""
    #     state, state_shapes = self._create_initial_state(initial_ids, initial_cache)
    #
    #     for _ in range(self.max_decode_length):
    #         state = self._search_step(state)
    #         finished_flags = state[_StateKeys.FINISHED_FLAGS]
    #         if finished_flags.float().sum() == self.batch_size * self.beam_size:
    #             break
    #
    #     alive_seq = state[_StateKeys.ALIVE_SEQ]
    #     alive_log_probs = state[_StateKeys.ALIVE_LOG_PROBS]
    #     finished_seq = state[_StateKeys.FINISHED_SEQ]
    #     finished_scores = state[_StateKeys.FINISHED_SCORES]
    #
    #     # Account for corner case where there are no finished sequences for a
    #     # particular batch item. In that case, return alive sequences for that batch
    #     # item.
    #     finished_seq = torch.where(
    #         finished_flags.any(dim=1, keepdim=True).repeat(1, self.beam_size, 1),
    #         finished_seq,
    #         alive_seq,
    #     )
    #     finished_scores = torch.where(
    #         finished_flags.any(dim=1, keepdim=True).repeat(1, self.beam_size),
    #         finished_scores,
    #         alive_log_probs,
    #     )
    #
    #     return finished_seq, finished_scores

    def _create_initial_state(self, initial_ids, initial_cache):
        """Return initial state dictionary and its shape invariants.

        Args:
          initial_ids: initial ids to pass into the symbols_to_logits_fn.
            int tensor with shape [batch_size, 1]
          initial_cache: dictionary storing values to be passed into the
            symbols_to_logits_fn.

        Returns:
            state and shape invariant dictionaries with keys from _StateKeys
        """
        for key, value in initial_cache.items():
            for inner_value in torch.flatten(value):
                if inner_value.dtype != self.dtype:
                    raise TypeError(
                        "initial_cache element for key '%s' has dtype %s that does not "
                        "match SequenceBeamSearch's dtype of %s. Value: %s" %
                        (key, value.dtype.name, self.dtype.name, inner_value))

        # Current loop index (starts at 0)
        cur_index = torch.zeros((), dtype=torch.int32)

        # Create alive sequence with shape [batch_size, beam_size, 1]
        alive_seq = _expand_to_beam_size(initial_ids, self.beam_size)
        alive_seq = torch.unsqueeze(alive_seq, axis=2)
        if self.padded_decode:
            alive_seq = alive_seq.repeat(1, 1, self.max_decode_length + 1)

        # Create tensor for storing initial log probabilities.
        # Assume initial_ids are prob 1.0
        initial_log_probs = torch.tensor(
            [[0.] + [-float("inf")] * (self.beam_size - 1)], dtype=self.dtype)
        alive_log_probs = initial_log_probs.repeat(self.batch_size, 1)

        # Expand all values stored in the dictionary to the beam size, so that each
        # beam has a separate cache.

        # assuming initial_cache is a dictionary
        alive_cache = map_structure(lambda t: _expand_to_beam_size(t, self.beam_size), initial_cache)

        # Initialize tensor storing finished sequences with filler values.
        finished_seq = torch.zeros_like(alive_seq)

        # Set scores of the initial finished seqs to negative infinity.
        finished_scores = torch.ones([self.batch_size, self.beam_size],
                                     dtype=self.dtype) * -inf(self.dtype)

        # Initialize finished flags with all False values.
        finished_flags = torch.zeros([self.batch_size, self.beam_size], dtype=torch.bool)

        # Create state dictionary
        state = {
            _StateKeys.CUR_INDEX: cur_index,
            _StateKeys.ALIVE_SEQ: alive_seq,
            _StateKeys.ALIVE_LOG_PROBS: alive_log_probs,
            _StateKeys.ALIVE_CACHE: alive_cache,
            _StateKeys.FINISHED_SEQ: finished_seq,
            _StateKeys.FINISHED_SCORES: finished_scores,
            _StateKeys.FINISHED_FLAGS: finished_flags
        }

        # Create state invariants for each value in the state dictionary. Each
        # dimension must be a constant or None. A None dimension means either:
        #   1) the dimension's value is a tensor that remains the same but may
        #      depend on the input sequence to the model (e.g. batch size).
        #   2) the dimension may have different values on different iterations.

        if self.padded_decode:
            state_shape_invariants = {
                _StateKeys.CUR_INDEX:
                    torch.TensorShape([]),
                _StateKeys.ALIVE_SEQ:
                    torch.TensorShape(
                        [self.batch_size, self.beam_size,
                         self.max_decode_length + 1]),
                _StateKeys.ALIVE_LOG_PROBS:
                    torch.TensorShape([self.batch_size, self.beam_size]),
                _StateKeys.ALIVE_CACHE:
                    map_structure(_get_shape, alive_cache),
                _StateKeys.FINISHED_SEQ:
                    torch.TensorShape(
                        [self.batch_size, self.beam_size,
                         self.max_decode_length + 1]),
                _StateKeys.FINISHED_SCORES:
                    torch.TensorShape([self.batch_size, self.beam_size]),
                _StateKeys.FINISHED_FLAGS:
                    torch.TensorShape([self.batch_size, self.beam_size])
            }
        else:
            state_shape_invariants = {
                _StateKeys.CUR_INDEX:
                    torch.TensorShape([]),
                _StateKeys.ALIVE_SEQ:
                    torch.TensorShape([None, self.beam_size, None]),
                _StateKeys.ALIVE_LOG_PROBS:
                    torch.TensorShape([None, self.beam_size]),
                _StateKeys.ALIVE_CACHE:
                    map_structure(_get_shape_keep_last_dim, alive_cache),
                _StateKeys.FINISHED_SEQ:
                    torch.TensorShape([None, self.beam_size, None]),
                _StateKeys.FINISHED_SCORES:
                    torch.TensorShape([None, self.beam_size]),
                _StateKeys.FINISHED_FLAGS:
                    torch.TensorShape([None, self.beam_size])
            }

        return state, state_shape_invariants

    def _continue_search(self, state):
        """Return whether to continue the search loop.

        The loops should terminate when
          1) when decode length has been reached, or
          2) when the worst score in the finished sequences is better than the best
             score in the alive sequences (i.e. the finished sequences are provably
             unchanging)

        Args:
          state: A dictionary with the current loop state.

        Returns:
          Bool tensor with value True if loop should continue, False if loop should
          terminate.
        """
        i = state[_StateKeys.CUR_INDEX]
        alive_log_probs = state[_StateKeys.ALIVE_LOG_PROBS]
        finished_scores = state[_StateKeys.FINISHED_SCORES]
        finished_flags = state[_StateKeys.FINISHED_FLAGS]

        not_at_max_decode_length = i < self.max_decode_length

        # Calculate largest length penalty (the larger penalty, the better score).
        max_length_norm = _length_normalization(self.alpha, self.max_decode_length,
                                                dtype=self.dtype)
        # Get the best possible scores from alive sequences.
        best_alive_scores = alive_log_probs[:, 0] / max_length_norm

        # Compute worst score in finished sequences for each batch element
        finished_scores *= finished_flags.type(self.dtype)  # set filler scores to zero
        lowest_finished_scores = torch.min(finished_scores, dim=1)[0]

        # If there are no finished sequences in a batch element, then set the lowest
        # finished score to -INF for that element.
        finished_batches = torch.any(finished_flags, dim=1)
        lowest_finished_scores += ((1.0 -
                                    finished_batches.type(self.dtype)) *
                                   -inf(self.dtype))

        worst_finished_score_better_than_best_alive_score = torch.all(
            lowest_finished_scores > best_alive_scores
        )

        return not_at_max_decode_length and not worst_finished_score_better_than_best_alive_score

    def _search_step(self, state):
        """Beam search loop body.

        Grow alive sequences by a single ID. Sequences that have reached the EOS
        token are marked as finished. The alive and finished sequences with the
        highest log probabilities and scores are returned.

        A sequence's finished score is calculating by dividing the log probability
        by the length normalization factor. Without length normalization, the
        search is more likely to return shorter sequences.

        Args:
          state: A dictionary with the current loop state.

        Returns:
          new state dictionary.
        """
        # Grow alive sequences by one token.
        new_seq, new_log_probs, new_cache = self._grow_alive_seq(state)
        # Collect top beam_size alive sequences
        alive_state = self._get_new_alive_state(new_seq, new_log_probs, new_cache)

        # Combine newly finished sequences with existing finished sequences, and
        # collect the top k scoring sequences.
        finished_state = self._get_new_finished_state(state, new_seq, new_log_probs)

        # Increment loop index and create new state dictionary
        new_state = {_StateKeys.CUR_INDEX: state[_StateKeys.CUR_INDEX] + 1}
        new_state.update(alive_state)
        new_state.update(finished_state)
        return [new_state]

    def _grow_alive_seq(self, state):
        """Grow alive sequences by one token, and collect top 2*beam_size sequences.

        2*beam_size sequences are collected because some sequences may have reached
        the EOS token. 2*beam_size ensures that at least beam_size sequences are
        still alive.

        Args:
          state: A dictionary with the current loop state.
        Returns:
          Tuple of
          (Top 2*beam_size sequences [batch_size, 2 * beam_size, cur_index + 1],
           Scores of returned sequences [batch_size, 2 * beam_size],
           New alive cache, for each of the 2 * beam_size sequences)
        """
        i = state[_StateKeys.CUR_INDEX]
        alive_seq = state[_StateKeys.ALIVE_SEQ]
        alive_log_probs = state[_StateKeys.ALIVE_LOG_PROBS]
        alive_cache = state[_StateKeys.ALIVE_CACHE]

        beams_to_keep = 2 * self.beam_size

        # Get logits for the next candidate IDs for the alive sequences. Get the new
        # cache values at the same time

        if self.padded_decode:
            flat_ids = alive_seq[:, :, i].reshape(self.batch_size * self.beam_size, 1)
        else:
            flat_ids = _flatten_beam_dim(alive_seq)  # [batch_size * beam_size]

        # Use pad_sequence to handle padding for cache
        flat_cache = [
            rnn_utils.pad_sequence([x[:, b, :] for b in range(self.beam_size)], batch_first=True, padding_value=0) for x
            in alive_cache]
        flat_logits, flat_cache = self.symbols_to_logits_fn(flat_ids, i, flat_cache)

        # Unflatten logits to shape [batch_size, beam_size, vocab_size]
        logits = _unflatten_beam_dim(flat_logits, self.batch_size, self.beam_size)
        new_cache = {k: _unflatten_beam_dim(v, self.batch_size, self.beam_size)
                     for k, v in flat_cache.items()}

        # Convert logits to normalized log probs
        candidate_log_probs = _log_prob_from_logits(logits)

        # Calculate new log probabilities if each of the alive sequences were
        # extended # by the the candidate IDs.
        # Shape [batch_size, beam_size, vocab_size]

        log_probs = candidate_log_probs + torch.unsqueeze(alive_log_probs, axis=2)

        # Each batch item has beam_size * vocab_size candidate sequences. For each
        # batch item, get the k candidates with the highest log probabilities.
        flat_log_probs = log_probs.reshape([-1, self.beam_size * self.vocab_size])
        topk_log_probs, topk_indices = torch.topk(flat_log_probs, k=beams_to_keep, dim=-1)

        # Extract the alive sequences that generate the highest log probabilities
        # after being extended.
        topk_beam_indices = topk_indices // self.vocab_size
        topk_seq, new_cache = _gather_beams(
            [alive_seq, new_cache], topk_beam_indices, self.batch_size,
            beams_to_keep)

        # Append the most probable IDs to the topk sequences
        topk_ids = topk_indices % self.vocab_size
        if self.padded_decode:
            topk_seq = topk_seq.transpose(2, 0, 1)
            topk_seq = topk_seq.scatter(0, i + 1, topk_ids)
            topk_seq = topk_seq.transpose([1, 2, 0])
        else:
            topk_ids = topk_ids.unsqueeze(2)
            topk_seq = torch.cat([topk_seq, topk_ids], dim=2)

        return topk_seq, topk_log_probs, new_cache

    def _get_new_alive_state(self, new_seq, new_log_probs, new_cache):
        """Gather the top k sequences that are still alive.

        Args:
          new_seq: New sequences generated by growing the current alive sequences
            int32 tensor with shape [batch_size, 2 * beam_size, cur_index + 1]
          new_log_probs: Log probabilities of new sequences
            float32 tensor with shape [batch_size, beam_size]
          new_cache: Dict of cached values for each sequence.

        Returns:
          Dictionary with alive keys from _StateKeys:
            {Top beam_size sequences that are still alive (don't end with eos_id)
             Log probabilities of top alive sequences
             Dict cache storing decoder states for top alive sequences}
        """
        # To prevent finished sequences from being considered, set log probs to -inf
        new_finished_flags = torch.eq(new_seq[:, :, -1], self.eos_id)
        new_log_probs += torch.cast(new_finished_flags, self.dtype) * float('-inf')

        # top_alive_log_probs, top_alive_indices = torch.topk(new_log_probs.view(self.batch_size, -1), k=self.beam_size,
        #                                                     dim=-1)
        # top_alive_seq = torch.zeros((self.batch_size, self.beam_size, new_seq.shape[-1]), dtype=torch.int32)
        # top_alive_cache = {}
        # for b in range(self.batch_size):
        #     for k in range(self.beam_size):
        #         idx = top_alive_indices[b, k] // self.vocab_size
        #         pos = top_alive_indices[b, k] % self.vocab_size
        #         top_alive_seq[b, k, :-1] = new_seq[b, idx, :]
        #         top_alive_seq[b, k, -1] = pos
        #         top_alive_cache[b, k] = {}
        #         for key, value in new_cache.items():
        #             if isinstance(value, dict):
        #                 top_alive_cache[b, k][key] = value[b, idx, :].unsqueeze(0)
        #             else:
        #                 top_alive_cache[b, k] = value[b, idx, :].unsqueeze(0)

        top_alive_seq, top_alive_log_probs, top_alive_cache = _gather_topk_beams(
            [new_seq, new_log_probs, new_cache], new_log_probs, self.batch_size,
            self.beam_size)

        return {
            _StateKeys.ALIVE_SEQ: top_alive_seq,
            _StateKeys.ALIVE_LOG_PROBS: top_alive_log_probs,
            _StateKeys.ALIVE_CACHE: top_alive_cache
        }

    def _get_new_finished_state(self, state, new_seq, new_log_probs):
        """Combine new and old finished sequences, and gather the top k sequences.

        Args:
          state: A dictionary with the current loop state.
          new_seq: New sequences generated by growing the current alive sequences
            int32 tensor with shape [batch_size, beam_size, i + 1]
          new_log_probs: Log probabilities of new sequences
            float32 tensor with shape [batch_size, beam_size]

        Returns:
          Dictionary with finished keys from _StateKeys:
            {Top beam_size finished sequences based on score,
             Scores of finished sequences,
             Finished flags of finished sequences}
        """
        i = state[_StateKeys.CUR_INDEX]
        finished_seq = state[_StateKeys.FINISHED_SEQ]
        finished_scores = state[_StateKeys.FINISHED_SCORES]
        finished_flags = state[_StateKeys.FINISHED_FLAGS]

        # First append a column of 0-ids to finished_seq to increment the length.
        # New shape of finished_seq: [batch_size, beam_size, i + 1]
        if not self.padded_decode:
            finished_seq = torch.cat([
                finished_seq,
                torch.zeros([self.batch_size, self.beam_size, 1], dtype=torch.int32)
            ],
                dim=2)

        # Calculate new seq scores from log probabilities.
        length_norm = _length_normalization(self.alpha, i + 1, dtype=self.dtype)
        new_scores = new_log_probs / length_norm

        # Set the scores of the still-alive seq in new_seq to large negative values.
        new_finished_flags = torch.eq(new_seq[:, :, -1], self.eos_id)
        new_scores += ((1. - new_finished_flags.to(dtype=self.dtype)) *
                       -inf(self.dtype))

        # Combine sequences, scores, and flags.
        finished_seq = torch.cat([finished_seq, new_seq], dim=1)
        finished_scores = torch.cat([finished_scores, new_scores], dim=1)
        finished_flags = torch.cat([finished_flags, new_finished_flags], dim=1)

        # Return the finished sequences with the best scores.
        top_finished_seq, top_finished_scores, top_finished_flags = (
            _gather_topk_beams([finished_seq, finished_scores, finished_flags],
                               finished_scores, self.batch_size, self.beam_size))

        return {
            _StateKeys.FINISHED_SEQ: top_finished_seq,
            _StateKeys.FINISHED_SCORES: top_finished_scores,
            _StateKeys.FINISHED_FLAGS: top_finished_flags
        }


def sequence_beam_search(
        symbols_to_logits_fn, initial_ids, initial_cache, vocab_size, beam_size,
        alpha, max_decode_length, eos_id, padded_decode=False):
    """Search for sequence of subtoken ids with the largest probability.

    Args:
      symbols_to_logits_fn: A function that takes in ids, index, and cache as
        arguments. The passed in arguments will have shape:
          ids -> A tensor with shape [batch_size * beam_size, index].
          index -> A scalar.
          cache -> A nested dictionary of tensors [batch_size * beam_size, ...].
        The function must return a tuple of logits and new cache:
          logits -> A tensor with shape [batch * beam_size, vocab_size].
          new cache -> A nested dictionary with the same shape/structure as the
            inputted cache.
      initial_ids: An int32 tensor with shape [batch_size]. Starting ids for
        each batch item.
      initial_cache: A dictionary, containing starting decoder variables
        information.
      vocab_size: An integer, the size of the vocabulary, used for topk
        computation.
      beam_size: An integer, the number of beams.
      alpha: A float, defining the strength of length normalization.
      max_decode_length: An integer, the maximum length to decoded a sequence.
      eos_id: An integer, ID of eos token, used to determine when a sequence has
        finished.
      padded_decode: A bool, indicating if max_sequence_length padding is used
        for beam search.

    Returns:
      Top decoded sequences [batch_size, beam_size, max_decode_length]
      sequence scores [batch_size, beam_size]
    """
    device = initial_ids.device
    batch_size = (
        initial_ids.shape[0] if padded_decode else
        initial_ids.size(0))
    sbs = SequenceBeamSearch(symbols_to_logits_fn, vocab_size, batch_size,
                             beam_size, alpha, max_decode_length, eos_id,
                             padded_decode)

    initial_ids = initial_ids.unsqueeze(1).repeat(1, beam_size).view(-1)
    initial_cache = {k: v.repeat(1, beam_size, 1).view(batch_size * beam_size, -1) for k, v in initial_cache.items()}

    ids, scores = sbs.search(initial_ids, initial_cache, device)
    ids = ids.view(batch_size, beam_size, -1)
    scores = scores.view(batch_size, beam_size)

    return ids, scores


def sequence_beam_search(
        symbols_to_logits_fn, initial_ids, initial_cache, vocab_size, beam_size,
        alpha, max_decode_length, eos_id, padded_decode=False):
    """Search for sequence of subtoken ids with the largest probability.

    Args:
      symbols_to_logits_fn: A function that takes in ids, index, and cache as
        arguments. The passed in arguments will have shape:
          ids -> A tensor with shape [batch_size * beam_size, index].
          index -> A scalar.
          cache -> A nested dictionary of tensors [batch_size * beam_size, ...].
        The function must return a tuple of logits and new cache:
          logits -> A tensor with shape [batch * beam_size, vocab_size].
          new cache -> A nested dictionary with the same shape/structure as the
            inputted cache.
      initial_ids: An int32 tensor with shape [batch_size]. Starting ids for
        each batch item.
      initial_cache: A dictionary, containing starting decoder variables
        information.
      vocab_size: An integer, the size of the vocabulary, used for topk
        computation.
      beam_size: An integer, the number of beams.
      alpha: A float, defining the strength of length normalization.
      max_decode_length: An integer, the maximum length to decoded a sequence.
      eos_id: An integer, ID of eos token, used to determine when a sequence has
        finished.
      padded_decode: A bool, indicating if max_sequence_length padding is used
        for beam search.

    Returns:
      Top decoded sequences [batch_size, beam_size, max_decode_length]
      sequence scores [batch_size, beam_size]
    """
    batch_size = (
        initial_ids.shape[0] if padded_decode else
        initial_ids.size()[0])
    sbs = SequenceBeamSearch(symbols_to_logits_fn, vocab_size, batch_size,
                             beam_size, alpha, max_decode_length, eos_id,
                             padded_decode)
    return sbs.search(initial_ids, initial_cache)


def _log_prob_from_logits(logits):
    return logits - torch.logsumexp(logits, dim=2, keepdim=True)


def _length_normalization(alpha, length, dtype=torch.float32):
    """Return length normalization factor."""
    return ((5. + length.to(dtype)) / 6.).pow(alpha)


def _expand_to_beam_size(tensor, beam_size):
    """Tiles a given tensor by beam_size.

    Args:
      tensor: tensor to tile [batch_size, ...]
      beam_size: How much to tile the tensor by.

    Returns:
      Tiled tensor [batch_size, beam_size, ...]
    """
    tensor = tensor.unsqueeze(1)
    tile_dims = [1] * tensor.dim()
    tile_dims[1] = beam_size

    return tensor.repeat(*tile_dims)


def _shape_list(tensor):
    """Return a list of the tensor's shape, and ensure no None values in list."""
    # Get statically known shape (may contain None's for unknown dimensions)
    shape = tensor.shape

    # Ensure that the shape values are not None
    dynamic_shape = tensor.size()
    for i in range(len(shape)):
        if shape[i] is None:
            shape[i] = dynamic_shape[i]
    return shape.tolist()


def _get_shape_keep_last_dim(tensor):
    shape_list = list(tensor.size())

    # Only the last
    for i in range(len(shape_list) - 1):
        shape_list[i] = None

    if isinstance(shape_list[-1], torch.Tensor):
        shape_list[-1] = None
    return torch.Size(shape_list)


def _get_shape(tensor):
    """Return the shape of the input tensor."""
    return tuple(tensor.shape)


def _flatten_beam_dim(tensor):
    """Reshapes first two dimensions in to single dimension.

    Args:
      tensor: Tensor to reshape of shape [A, B, ...]

    Returns:
      Reshaped tensor of shape [A*B, ...]
    """
    shape = list(tensor.shape)
    shape[0] *= shape[1]
    shape.pop(1)  # Remove beam dim
    return tensor.reshape(shape)


def _unflatten_beam_dim(tensor, batch_size, beam_size):
    """Reshapes first dimension back to [batch_size, beam_size].

    Args:
      tensor: Tensor to reshape of shape [batch_size*beam_size, ...]
      batch_size: Tensor, original batch size.
      beam_size: int, original beam size.

    Returns:
      Reshaped tensor of shape [batch_size, beam_size, ...]
    """
    shape = list(tensor.shape)
    new_shape = [batch_size, beam_size] + shape[1:]
    return tensor.reshape(new_shape)


def _gather_beams(nested, beam_indices, batch_size, new_beam_size):
    """Gather beams from nested structure of tensors.

    Each tensor in nested represents a batch of beams, where beam refers to a
    single search state (beam search involves searching through multiple states
    in parallel).

    This function is used to gather the top beams, specified by
    beam_indices, from the nested tensors.

    Args:
      nested: Nested structure (tensor, list, tuple or dict) containing tensors
        with shape [batch_size, beam_size, ...].
      beam_indices: int32 tensor with shape [batch_size, new_beam_size]. Each
       value in beam_indices must be between [0, beam_size), and are not
       necessarily unique.
      batch_size: int size of batch
      new_beam_size: int number of beams to be pulled from the nested tensors.

    Returns:
      Nested structure containing tensors with shape
        [batch_size, new_beam_size, ...]
    """
    # Computes the i'th coodinate that contains the batch index for gather_nd.
    # Batch pos is a tensor like [[0,0,0,0,],[1,1,1,1],..].
    batch_pos = torch.arange(batch_size * new_beam_size) // new_beam_size
    batch_pos = batch_pos.view(batch_size, new_beam_size)

    # Create coordinates to be passed to torch.gather. Stacking creates a tensor
    # with shape [batch_size, beam_size, 2], where the last dimension contains
    # the (i, j) gathering coordinates.
    coordinates = torch.stack([batch_pos, beam_indices], dim=2)

    return map_structure(
        lambda state: torch.gather(state, 1, coordinates.expand(-1, -1, *state.shape[2:])),
        nested)


def _gather_topk_beams(nested, score_or_log_prob, batch_size, beam_size):
    """Gather top beams from nested structure."""
    _, topk_indexes = torch.topk(score_or_log_prob, k=beam_size, dim=-1)
    return _gather_beams(nested, topk_indexes, batch_size, beam_size)
