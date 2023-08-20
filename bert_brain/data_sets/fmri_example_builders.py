import dataclasses
from typing import Sequence, Any, Optional

import numpy as np


__all__ = ['FMRICombinedSentenceExamples', 'FMRIExample', 'PairFMRIExample']


@dataclasses.dataclass
class FMRIExample:
    words: Sequence[Any]
    sentence_ids: Sequence[int]
    tr_target: Sequence[Optional[Sequence[int]]]
    full_sentences: Sequence[Any]
    offset: int


@dataclasses.dataclass
class PairFMRIExample(FMRIExample):
    second_offset: int
    len_1: int
    len_1_full: int


class FMRICombinedSentenceExamples:

    def __init__(
            self,
            window_duration: float = 8.,
            minimum_duration_required: float = 7.8,
            use_word_unit_durations: bool = False,
            sentence_mode: str = 'multiple'):
        """
        For each TR, finds the minimal combination of sentences that will give window_size_features seconds of data
        for the TR. For example, if the word timings are

        [('The', 0.5), ('dog', 1.0), ('chased', 1.5), ('the', 2.0), ('cat', 2.5), ('up', 3.0), ('the', 3.5),
         ('tree.', 4.0), ('The', 4.5), ('cat', 5.0), ('was', 5.5), ('scared', 6.0)]

        and if the window_size_features is 2.0, then for the TR at ('cat', 2.5), the combination of sentences
        that gives 2.0 seconds of features is simply the first sentence: (0,)
        For the TR at ('cat', 5.0), the combination is: (0, 1)

        Once these combinations have been computed, some of them will be subsets of others. The function removes
        any combinations which are 