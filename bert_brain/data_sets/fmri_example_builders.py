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


class FMRICombinedSentence