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
                tokens will be used as the example_key. However, this may be unde