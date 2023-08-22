
import os
from collections import OrderedDict
from itertools import combinations
from dataclasses import dataclass, replace as dataclass_replace
from typing import Mapping, Sequence, Optional, Union
from functools import partial

import numpy as np
from scipy.spatial.distance import cdist
from scipy.io import loadmat
from scipy.ndimage.filters import gaussian_filter

import nibabel
import cortex

from ..common import MultiReplace
from .corpus_base import CorpusBase, CorpusExampleUnifier
from .fmri_example_builders import FMRICombinedSentenceExamples, FMRIExample, PairFMRIExample
from .input_features import RawData, KindData, ResponseKind


__all__ = ['HarryPotterCorpus', 'read_harry_potter_story_features', 'harry_potter_leave_out_fmri_run',
           'HarryPotterMakeLeaveOutFmriRun', 'get_indices_from_normalized_coordinates', 'get_mask_for_subject']


@dataclass
class _HarryPotterWordFMRI:
    word: str
    index_in_all_words: int
    index_in_sentence: int
    sentence_id: int
    time: float
    run: int
    story_features: Mapping


@dataclass(frozen=True)
class _MEGKindProperties:
    file_name: str
    is_preprocessed: bool = False


class HarryPotterCorpus(CorpusBase):

    @classmethod
    def _path_attributes(cls):
        return dict(path='harry_potter_path')

    # we need to use these so we can have run information even when we
    # don't read in fmri data; i.e. as a way to do train-test splits the
    # same for MEG as for fMRI. We will assert that the run lengths we
    # get are equal to these when we read fMRI
    static_run_lengths = (340, 352, 279, 380)

    def __init__(
            self,
            path: Optional[str] = None,
            meg_subjects: Optional[Sequence[str]] = None,
            meg_kind: str = 'pca',
            separate_meg_axes: Optional[Union[str, Sequence[str]]] = None,
            group_meg_sentences_like_fmri: bool = False,
            fmri_subjects: Optional[Sequence[str]] = None,
            fmri_smooth_factor: Optional[float] = 1.,
            fmri_skip_start_trs: int = 20,
            fmri_skip_end_trs: int = 15,
            fmri_window_duration: float = 8.,
            fmri_minimum_duration_required: float = 7.8,
            fmri_sentence_mode: str = 'multiple'):
        """
        Loader for Harry Potter data
        Args:
            path: The path to the directory where the data is stored
            meg_subjects: Which subjects' data to load for MEG. None will cause all subjects' data to load. An
                empty list can be provided to cause MEG loading to be skipped.
            meg_kind: One of ('pca_label', 'mean_label', 'pca_sensor')
                pca_label is source localized and uses the pca within an ROI label for the label with
                    100ms slices of time
                mean_label is similar to pca_label, but using the mean within an ROI label
                pca_sensor uses a PCA on the sensor space to produce 10 latent components and time is averaged over
                the whole word
            separate_meg_axes: None, True, or one or more of ('subject', 'roi', 'time'). If not provided (the default),
                then when MEG data has not been preprocessed, a dictionary with a single key ('hp_meg') that maps to
                an array of shape (word, subject, roi, 100ms slice) [Note that this shape may be modified by
                preprocessing] is returned. 'subject', 'roi', and 'time' only apply to non-preprocessed data.
                If 'subject' is provided, then the dictionary is keyed by 'hp_meg.<subject-id>', e.g. 'hp_meg.A",
                and the shape of each value is (word, roi, 100ms slice). Similarly each separate axis that is
                provided causes the data array to be further split and each resulting data array can be found under a
                more complex key. The keys are generated in the order <subject-id>.<roi>.<time> all of which are
                optional. When time is in the key, it is a multiple of 100 giving the ms start time of the window. If
                the meg_kind is a preprocessed_kind, then only True is supported for this argument. If True, the
                data is split such that each component gets a separate key. For example, 'hp_meg_A.0', 'hp_meg_A.1', ...
                which is a scalar value for each word.
            group_meg_sentences_like_fmri: If False, examples for MEG are one sentence each. If True, then examples
                are created as they would be for fMRI, i.e. including sentences as required by the
                fmri_window_size_features parameter
            fmri_subjects: Which subjects' data to load for fMRI. None will cause all subjects' data to load. An
                empty list can be provided to cause fMRI loading to be skipped.
            fmri_smooth_factor: The sigma parameter of the gaussian blur function
                applied to blur the fMRI data spatially, or None to skip blurring.
            fmri_skip_start_trs: The number of TRs to remove from the beginning of each fMRI run, since the first few
                TRs can be problematic
            fmri_skip_end_trs: The number of TRs to remove from the end of each fMRI run, since the last few TRs can be
                problematic
            fmri_window_duration: The duration of the window of time preceding a TR from which to
                choose the words that will be involved in predicting that TR. For example, if this is 8, then all words
                which occurred with tr_time > word_time >= tr_time - 8 will be used to build the example for the TR.
            fmri_minimum_duration_required: The minimum duration of the time between the earliest word used to
                predict a TR and the occurrence of the TR. This much time is required for the TR to be a legitimate
                target. For example, if this is set to 7.5, then letting the time of the earliest word occurring in the
                window_duration before the TR be min_word_time, if tr_time - min_word_time <
                minimum_duration_required, the TR is not is not used to build any examples.
            fmri_sentence_mode: One of ['multiple', 'single', 'ignore']. When 'multiple', an example consists of the
                combination of sentences as described above. If 'single', changes the behavior of the function so that
                the feature window is truncated by the start of a sentence, thus resulting in examples with one
                sentence at a time. If 'ignore', then each example consists of exactly the words in the feature window
                without consideration of the sentence boundaries
        """
        self.path = path
        self.fmri_subjects = fmri_subjects
        self.meg_subjects = meg_subjects
        self.meg_kind = meg_kind
        self.separate_meg_axes = separate_meg_axes
        self.group_meg_sentences_like_fmri = group_meg_sentences_like_fmri
        self.fmri_smooth_factor = fmri_smooth_factor
        self.fmri_skip_start_trs = fmri_skip_start_trs
        self.fmri_skip_end_trs = fmri_skip_end_trs
        self.fmri_example_builder = FMRICombinedSentenceExamples(
            window_duration=fmri_window_duration,
            minimum_duration_required=fmri_minimum_duration_required,
            use_word_unit_durations=False,  # since the word-spacing is constant in Harry Potter, not needed
            sentence_mode=fmri_sentence_mode)

    @staticmethod
    def _add_fmri_example(
            example, example_manager: CorpusExampleUnifier, data_keys=None, data_ids=None,
            is_apply_data_id_to_entire_group=False, allow_new_examples=True, words_override=None):
        key = tuple(w.index_in_all_words for w in example.words)
        if isinstance(example, PairFMRIExample):
            len_1 = example.len_1
            start_2 = example.second_offset
            stop_2 = len(example.words) - len_1 + start_2
        else:
            len_1 = len(example.words)
            start_2 = None
            stop_2 = None
        features = example_manager.add_example(
            key,
            words_override if words_override is not None else [w.word for w in example.full_sentences],
            [w.sentence_id for w in example.full_sentences],
            data_keys,
            data_ids,
            start=example.offset,
            stop=example.offset + len_1,
            start_sequence_2=start_2,
            stop_sequence_2=stop_2,
            is_apply_data_id_to_entire_group=is_apply_data_id_to_entire_group,
            allow_new_examples=allow_new_examples)

        assert (all(w.run == example.words[0].run for w in example.words[1:]))
        return features

    def story_features_per_fmri_example(self, paths_obj):
        if paths_obj is not None:
            self.set_paths_from_path_object(path_obj=paths_obj)
        else:
            self.check_paths()
        fmri_examples = self._compute_examples_for_fmri()
        unique_ids = dict()
        words = dict()
        for example in fmri_examples:
            key = tuple(w.index_in_all_words for w in example.words)
            if key not in unique_ids:
                unique_ids[key] = len(unique_ids)
            words[unique_ids[key]] = example.words
        return words

    @staticmethod
    def _meg_kind_properties(meg_kind):
        kind_properties = {
            'pca_label': _MEGKindProperties('harry_potter_meg_100ms_pca.npz'),
            'mean_label': _MEGKindProperties('harry_potter_meg_100ms_mean_flip.npz'),
            'pca_sensor': _MEGKindProperties('harry_potter_meg_sensor_pca_35_word_mean.npz'),
            'pca_sensor_full': _MEGKindProperties('harry_potter_meg_sensor_pca_35_word_full.npz'),
            'ica_sensor_full': _MEGKindProperties('harry_potter_meg_sensor_ica_35_word_full.npz'),
            'leila': _MEGKindProperties('harry_potter_meg_sensor_25ms_leila.npz'),
            'rank_clustered_kmeans_L2_A': _MEGKindProperties(
                'harry_potter_meg_rank_clustered_kmeans_L2_A.npz', is_preprocessed=True),
            'rank_clustered_kmeans': _MEGKindProperties(
                'harry_potter_meg_rank_clustered_kmeans_L2.npz', is_preprocessed=True),
            'rank_clustered_L2': _MEGKindProperties(
                'harry_potter_meg_rank_clustered_L2.npz', is_preprocessed=True),
            'rank_clustered_median': _MEGKindProperties(
                'harry_potter_meg_rank_clustered_median.npz', is_preprocessed=True),
            'rank_clustered_mean': _MEGKindProperties(
                'harry_potter_meg_rank_clustered_mean.npz', is_preprocessed=True),
            'rank_clustered_rms': _MEGKindProperties(
                'harry_potter_meg_rank_clustered_rms.npz', is_preprocessed=True),
            'rank_clustered_counts': _MEGKindProperties(
                'harry_potter_meg_rank_clustered_counts.npz', is_preprocessed=True),
            'rank_clustered_mean_time_slice_ms_100_A': _MEGKindProperties(
                'harry_potter_meg_rank_clustered_mean_time_slice_ms_100_A.npz', is_preprocessed=True),
            'rank_clustered_mean_whole_A': _MEGKindProperties(
                'harry_potter_meg_rank_clustered_mean_whole_A.npz', is_preprocessed=True),
            'rank_clustered_sum_time_slice_ms_100_A': _MEGKindProperties(
                'harry_potter_meg_rank_clustered_sum_time_slice_ms_100_A.npz', is_preprocessed=True),
            'direct_rank_clustered_sum_25_ms': _MEGKindProperties(
                'harry_potter_meg_direct_rank_clustered_sum_25.npz', is_preprocessed=True),
        }

        if meg_kind not in kind_properties:
            raise ValueError('Unknown meg_kind: {}'.format(meg_kind))

        return kind_properties[meg_kind]

    @property
    def meg_path(self):
        return os.path.join(self.path, HarryPotterCorpus._meg_kind_properties(self.meg_kind).file_name)

    def _run_info(self, index_run):
        if self.meg_kind in ('rank_clustered',):
            return index_run % 4
        return -1

    def _load(self, run_info, example_manager: CorpusExampleUnifier):

        data = OrderedDict()

        run_at_unique_id = list()

        fmri_examples = None
        if (((self.meg_subjects is None or len(self.meg_subjects) > 0) and self.group_meg_sentences_like_fmri)
                or self.fmri_subjects is None  # None means all subjects
                or len(self.fmri_subjects)) > 0:
            fmri_examples = self._compute_examples_for_fmri()
            for example in fmri_examples:
                features = HarryPotterCorpus._add_fmri_example(example, example_manager)
                assert(features.unique_id == len(run_at_unique_id))
                run_at_unique_id.append(example.words[0].run)
        meg_examples = None
        if self.meg_subjects is None or len(self.meg_subjects) > 0:
            if not self.group_meg_sentences_like_fmri:
                # add all of the sentences first to guarantee consistent example ids
                sentences, _ = self._harry_potter_fmri_word_info(HarryPotterCorpus.static_run_lengths)
                meg_examples = list()
                for sentence_id, sentence in enumerate(sentences):
                    key = tuple(w.index_in_all_words for w in sentence)
                    features = example_manager.add_example(
                        key,
                        [w.word for w in sentence],
                        [w.sentence_id for w in sentence],
                        None,
                        None)
                    assert(all(w.run == sentence[0].run for w in sentence[1:]))
                    assert(features.unique_id <= len(run_at_unique_id))
                    if features.unique_id < len(run_at_unique_id):
                        assert(run_at_unique_id[features.unique_id] == sentence[0].run)
                    else:
                        run_at_unique_id.append(sentence[0].run)
                    # tr_target is not going to be used, so we don't actually build it out
                    meg_examples.append(FMRIExample(sentence, [sentence_id] * len(sentence), [], sentence, 0))
            else:
                meg_examples = fmri_examples

        run_at_unique_id = np.array(run_at_unique_id)
        run_at_unique_id.setflags(write=False)

        metadata = dict(fmri_runs=run_at_unique_id)

        indicator_validation = None
        if self.meg_subjects is None or len(self.meg_subjects) > 0:
            meg, block_metadata, indicator_validation = self._read_meg(run_info, example_manager, meg_examples)
            for k in meg:
                data[k] = KindData(ResponseKind.hp_meg, meg[k])
            if block_metadata is not None:
                metadata['meg_blocks'] = block_metadata
        if self.fmri_subjects is None or len(self.fmri_subjects) > 0:
            fmri = self._read_fmri(run_info, example_manager, fmri_examples)
            for k in fmri:
                data[k] = KindData(ResponseKind.hp_fmri, fmri[k])

        for k in data:
            data[k].data.setflags(write=False)

        input_examples = list(example_manager.iterate_examples(fill_data_keys=True))

        if indicator_validation is not None:
            train = list()
            validation = list()
            for is_validation, ex in zip(indicator_validation, input_examples):
                if is_validation:
                    validation.append(ex)
                else:
                    train.append(ex)
            return RawData(
                input_examples=train,
                validation_input_examples=validation, response_data=data, is_pre_split=True, metadata=metadata)

        return RawData(
            input_examples,
            response_data=data, metadata=metadata, validation_proportion_of_train=0.1, test_proportion=0.)

    def _read_preprocessed_meg(self, run_info, example_manager: CorpusExampleUnifier, examples):
        # see make_harry_potter.ipynb for how these are constructed

        with np.load(self.meg_path, allow_pickle=True) as loaded:
            blocks = loaded['blocks']
            stimuli = loaded['stimuli']
            assert (stimuli[2364] == '..."')
            stimuli[2364] = '...."'  # this was an ellipsis followed by a ., but the period got dropped somehow

            held_out_block = np.unique(blocks)[run_info]
            not_fixation = np.logical_not(stimuli == '+')
            new_indices = np.full(len(not_fixation), -1, dtype=np.int64)
            new_indices[not_fixation] = np.arange(np.count_nonzero(not_fixation))
            stimuli = stimuli[not_fixation]
            blocks = blocks[not_fixation]

            subjects = loaded['subjects']

            if self.meg_subjects is not None:
                indicator_subjects = np.array([s in self.meg_subjects for s in subjects])
                # noinspection PyTypeChecker
                subjects = subjects[indicator_subjects]

            data = OrderedDict()
            for subject in subjects:
                data['hp_meg_{}'.format(subject)] = \
                    loaded['data_{}_hold_out_{}'.format(subject, held_out_block)][not_fixation]

            if 'data_multi_subject_hold_out_{}'.format(held_out_block) in loaded \
                    and (self.meg_subjects is None or 'multi_subject' in self.meg_subjects):
                data['hp_meg_multi_subject'] = \
                    loaded['data_multi_subject_hold_out_{}'.format(held_out_block)][not_fixation]

        if self.separate_meg_axes is not None:
            if not isinstance(self.separate_meg_axes, bool):
                raise ValueError('Only boolean values are supported for \'separate_meg_axes\' '
                                 'when the meg_kind is preprocessed')
            if self.separate_meg_axes:
                separated_data = OrderedDict()
                for k in data:
                    assert(len(data[k].shape) == 2)
                    for index_task, item in enumerate(np.split(data[k], data[k].shape[1], axis=1)):
                        separated_data['{}.{}'.format(k, index_task)] = item
                data = separated_data

        block_metadata = np.full(len(example_manager), -1, dtype=np.int64)

        for example in examples:
            indices = np.array([w.index_in_all_words for w in example.full_sentences])
            # use these instead of the words given in example to ensure our indexing is not off
            # we will fail an assert below if we try to add something messed up
            example_stimuli = stimuli[new_indices[indices]]

            features = HarryPotterCorpus._add_fmri_example(
                example,
                example_manager,
                words_override=[_clean_word(w) for w in example_stimuli],
                data_keys=[k for k in data],
                data_ids=new_indices[indices],
                allow_new_examples=False)