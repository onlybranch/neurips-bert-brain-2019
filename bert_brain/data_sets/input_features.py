from dataclasses import dataclass
import dataclasses
from typing import Sequence, Optional, Mapping, Any, Union

import numpy as np
import torch


__all__ = ['InputFeatures', 'RawData', 'FieldSpec', 'KindData', 'ResponseKind', 'split_data']


@dataclass
class FieldSpec:
    fill_value: Any = None
    tensor_dtype: Any = torch.float
    is_sequence: bool = True

    def __post_init__(self):
        if self.fill_value is None:
            if self.tensor_dtype.is_floating_point:
                self.fill_value = np.nan
            else:
                self.fill_value = 0

    def __eq__(self, other):
        return np.isclose(self.fill_value, other.fill_value, rtol=0, atol=0, equal_nan=True) \
               and self.tensor_dtype == other.tensor_dtype \
               and self.is_sequence == other.is_sequence


@dataclass
class InputFeatures:
    unique_id: int
    tokens: Sequence[str]
    token_ids: Sequence[int]
    mask: Sequence[int]
    is_stop: Sequence[int]
    is_begin_word_pieces: Sequence[int]
    token_lengths: Sequence[int]
    token_probabilities: Sequence[float]
    type_ids: Sequence[int]
    head_location: Sequence[int]
    head_tokens: Sequence[str]
    head_token_ids: Sequence[int]
    index_word_in_example: Sequence[int]  # useful for grouping tokens together in the model
    index_token_in_sentence: Sequence[int]  # useful for positional embedding
    data_ids: Union[Mapping[str, Seque