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
        any combinations which are subsets of other combinations.

        The output is a sequence of examples. Each example has three fields:
            words: The sequence of 'words' associated with this example that are selected from the words passed in.
                Each word can be of any type, the function does not look at their values
            sentence_ids: A sequence of sentence ids, one for each word
            tr_target: A sequence of sequences of TR indices, one sequence for each word. If a word does not have
                a tr_target associated with it, then None replaces the sequence of TR indices. Multiple tr_targets
                can be associated with a single word depending on the timings and parameters of the function. A TR
                becomes the target for the final word in the window selected according to that TR's time.
            full_sentences: Gives the sequence of 'words' for the full sentences of which this example is a portion.
                When sentence_mode == 'multiple' or sentence_mode == 'single', this is the same sequence as words.
                When sentence_mode == 'ignore' the sequence may be different.
            offset: The offset from the beginning of full_sentences to the beginning of words. When
                sentence_mode == 'multiple' or sentence_mode == 'single', this is 0. When sentence_mode == 'ignore'
                this may be non-zero

        Args:
            window_duration: The duration of the window of time preceding a TR from which to
                choose the words that will be involved in predicting that TR. For example, if this is 8, then all words
                which occurred with tr_time > word_time >= tr_time - 8 will be used to build the example for the TR.
            minimum_duration_required: The minimum duration of the time between the earliest word used to
                predict a TR and the occurrence of the TR. This much time is required for the TR to be a legitimate
                target. For example, if this is set to 7.5, then letting the time of the earliest word occurring in the
                window_duration before the TR be min_word_time, if tr_time - min_word_time <
                minimum_duration_required, the TR is not is not used to build any examples.
            use_word_unit_durations: If True, then window_duration and minimum_duration_required are in number
                of words rather than time_units. window_duration = 8. would select the 8 previous words.
            sentence_mode: One of ['multiple', 'single', 'ignore']. When 'multiple', an example consists of the
                combination of sentences as described above. If 'single', changes the behavior of the function so that
                the feature window is truncated by the start of a sentence, thus resulting in examples with one
                sentence at a time. If 'ignore', then each example consists of exactly the words in the feature window
                without consideration of the sentence boundaries
        """
        self.window_duration = window_duration
        self.minimum_duration_required = minimum_duration_required
        self.use_word_unit_durations = use_word_unit_durations
        self.sentence_mode = sentence_mode

    def __call__(self, words, word_times, word_sentence_ids, tr_times, tr_offset=0):
        """
        For each TR, finds the minimal combination of sentences that will give window_size_features seconds of data
        for the TR. For example, if the word timings are

        [('The', 0.5), ('dog', 1.0), ('chased', 1.5), ('the', 2.0), ('cat', 2.5), ('up', 3.0), ('the', 3.5),
         ('tree.', 4.0), ('The', 4.5), ('cat', 5.0), ('was', 5.5), ('scared', 6.0)]

        and if the window_size_features is 2.0, then for the TR at ('cat', 2.5), the combination of sentences
        that gives 2.0 seconds of features is simply the first sentence: (0,)
        For the TR at ('cat', 5.0), the combination is: (0, 1)

        Once these combinations have been computed, some of them will be subsets of others. The function removes
        any combinations which are subsets of other combinations.

        Args:
            words: A list of 'words'. Each word can be of any type. Sequences of these are returned in each example,
                but they are otherwise unused by the function
            word_times: The time for each word, in the same time units as tr_times.
            word_sentence_ids: The sentence id for each word.
            tr_times: The time for each TR, in the same time units as word_times
            tr_offset: Added to the index of the target_trs. Useful when making multiple calls to this function on
                subsets of the TRs
        Returns:
            A sequence of examples. Each example has three fields:
            words: The sequence of 'words' associated with this example that are selected from the words passed in.
                Each word can be of any type, the function does not look at their values
            sentence_ids: A sequence of sentence ids, one for each word
            tr_target: A sequence of sequences of TR indices, one sequence for each word. If a word does not have
                a tr_target associated with it, then None replaces the sequence of TR indices. Multiple tr_targets
                can be associated with a single word depending on the timings and parameters of the function. A TR
                becomes the target for the final word in the window selected according to that TR's time.

        """
        word_times = np.asarray(word_times)
        if not np.all(np.diff(word_times) >= 0):
            raise ValueError('word_times must be monotonically increasing')
        word_sentence_ids = np.asarray(word_sentence_ids)
   