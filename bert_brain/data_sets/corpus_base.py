import os
from inspect import signature
import hashlib
from collections import OrderedDict
import itertools
import dataclasses
from typing import Sequence, Union, Optional, Hashable, Mapping

import numpy as np
import torch

from spacy.language import Language as SpacyLanguage
from pytorch_pretrained_bert import BertTokenizer

from .spacy_token_meta import bert_tokenize_with_spacy_meta
from .input_features import InputFeatures, FieldSpec
from .corpus_cache import save_to_cache, load_from_cache


__all__ = ['CorpusBase', 'CorpusExampleUnifier']


class CorpusExampleUnifier:

    def __init__(self, spacy_tokenize_model: SpacyLanguage, bert_tokenizer: BertTokenizer):
        self.spacy_tokenize_model = spacy_tokenize_model
        self.bert_tokenizer = bert_tokenizer
        self._examples = OrderedDict()
        self._seen_data_keys = OrderedDict()

    def add_example(
            self,
            example_key: Optional[Hashable],
            words: Sequence[str],
            sentence_ids: Sequence[int],
            data_key: Optional[Union[str, Sequence[str]]],
            data_ids: Optional[Sequence[int]],
            start: int = 0,
            stop: Optional[int] = None,
            start_sequence_2: Optional[int] = None,
            stop_sequence_2: Optional[int] = None,
            start_sequence_3: Optional[int] = None,
            stop_sequence_3: Optional[int] = None,
            is_apply_data_id_to_entire_group: bool = False,
            multipart_id: Optional[int] = None,
            span_ids: Optional[Sequence[int]] = None,
            allow_new_examples: bool = True) -> Optional[InputFeatures]:
        """
        Adds an example for the current data loader to return later. Simplifies the process of merging examples
        across different response measures. For example MEG and fMRI
        Args:
            example_key: For instance, the position of the example within a story. If this is set to None, then the
                tokens will be used as the example_key. However, this may be undesirable since in a given passage,
                sentences, especially short sentences, can be repeated.
            words: The words in the example
            sentence_ids: For each word, identifies which sentence the word belongs to. Used to compute
                index_of_word_in_sentence in the resulting InputFeatures
            data_key: A key (or multiple keys) to designate which response data set(s) data_ids references
            data_ids: indices into the response data, one for each token
            start: Offset where the actual input features should start. It is best to compute spacy meta on full
                sentences, then slice the resulting tokens. start and stop are used to slice words, sentence_ids,
                data_ids and type_ids
            stop: Exclusive end point for the actual input features. If None, the full length is used
            is_apply_data_id_to_entire_group: If a word is broken into multiple tokens, generally a single token is
                heuristically chosen as the 'main' token corresponding to that word. The data_id it is assigned is given
                by data offset, while all the tokens that are not the main token in the group are assigned -1. If this
                parameter is set to True, then all of the multiple tokens corresponding to a word are assigned the same
                data_id, and none are set to -1. This can be a better option for fMRI where the predictions are not at
                the word level, but rather at the level of an image containing multiple words.
            start_sequence_2: Used for bert to combine multiple sequences as a single input. Generally this is used for
                tasks like question answering where type_id=0 is the question and type_id=1 is the answer.
                If not specified, type_id=0 is used for every token
            stop_sequence_2: Used for bert to combine multiple sequences as a single input. Generally this is used for
                tasks like question answering where type_id=0 is the question and type_id=1 is the answer.
            start_sequence_3: Used for bert to combine 3 sequences as a single input. Generally this is used for tasks
                like question answering with a context. type_id=0 is the context and type_id=1 is the question and
                answer
            stop_sequence_3: Used for bert to combine 3 sequences as a single input. Generally this is used for tasks
                like question answering with a context. type_id=0 is the context and type_id=1 is the question and
                answer
            multipart_id: Used to express that this example needs to be in the same batch as other examples sharing the
                same multipart_id to be evaluated
            span_ids: Bit-encoded span identifiers which indicate which spans each word belongs to when spans are
                labeled in the input. If not given, no span ids will be set on the returned InputFeatures instance.
            allow_new_examples: If False, then if the example does not already exist in this instance, it will not
                be added. Only new data_ids will be added to existing examples. Returns None when the example does
                not exist.
        Returns:
            The InputFeatures instance associated with the example
        """
        input_features = bert_tokenize_with_spacy_meta(
            self.spacy_tokenize_model, self.bert_tokenizer,
            len(self._examples), words, sentence_ids, data_key, data_ids,
            start, stop,
            start_sequence_2, stop_sequence_2,
            start_sequence_3, stop_sequence_3,
            multipart_id,
            span_ids,
            is_apply_data_id_to_entire_group)

        if example_key is None:
            example_key = tuple(input_features.token_ids)

        if example_key not in self._examples:
            if allow_new_examples:
                self._examples[example_key] = input_features
            else:
                return None
        else:
            current = dataclasses.asdict(input_features)
            have = dataclasses.asdict(self._examples[example_key])
            assert(len(have) == len(current))
            for k in have:
                assert(k in current)
                if k == 'unique_id' or k == 'data_ids':
                    continue
                else:
                    # handles NaN, whereas np.array_equal does not
                    np.testing.assert_array_equal(have[k], current[k])
            if data_key is not None:
                if isinstance(data_key, str):
                    data_key = [data_key]
                for k in data_key:
                    self._seen_data_keys[k] = True
                    self._examples[example_key].data_ids[k] = input_features.data_ids[k]

        return self._examples[example_key]

    def iterate_examples(self, fill_data_keys=False):
        for k in self._examples:
            if fill_data_keys:
                for data_key in self._seen_data_keys:
                    if data_key not in self._examples[k].data_ids:
                        self._examples[k].data_ids[data_key] = -1 * np.ones(
                            len(self._examples[k].token_ids), dtype=np.int64)
            yield self._examples[k]

    def remove_data_keys(self, data_keys):
        if isinstance(data_keys, str):
            data_keys = [data_keys]
        for ex in self.iterate_examples():
            for data_key in data_keys:
                if data_key in ex.data_ids:
                    del ex.data_ids[data_key]
        for data_key in data_keys:
            if data_key in self._seen_data_keys:
                del self._seen_data_keys[data_key]

    def __len__(self):
        return len(self._examples)


class CorpusBase:

    @classmethod
    def _path_a