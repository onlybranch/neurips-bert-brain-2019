import hashlib
import itertools
import random

import numpy as np
import torch

from .common import SwitchRemember
from .data_sets import ResponseKind, CorpusTypes, PreprocessDetrend, PreprocessStandardize
from .modeling import KeyedLinear
from .settings import TrainingVariation, LoadFrom, Settings, OptimizationSettings, PredictionHeadSettings

__all__ = ['task_hash', 'set_random_seeds', 'iterate_powerset', 'named_variations', 'match_variation']


def _internal_hash_update(hash_, loss_tasks):
    if isinstance(loss_tasks, TrainingVariation):
        for loss_task in sorted(loss_tasks.loss_tasks):
            hash_.update(loss_task.encode())
        if loss_tasks.load_from is not None:
            hash_.update(loss_tasks.load_from.variation_name.encode())
            _internal_hash_update(hash_, loss_tasks.load_from.loss_tasks)
    else:
        for loss_task in sorted(loss_tasks):
            hash_.update(loss_task.encode())


def task_hash(loss_tasks):
    hash_ = hashlib.sha256()
    _internal_hash_update(hash_, loss_tasks)
    return hash_.hexdigest()


def set_random_seeds(seed, index_run, n_gpu):
    hash_ = hashlib.sha256('{}'.format(seed).encode())
    hash_.update('{}'.format(index_run).encode())
    seed = np.frombuffer(hash_.digest(), dtype='uint32')
    random_state = np.random.RandomState(seed)
    np.random.set_state(random_state.get_state())
    seed = np.random.randint(low=0, high=np.iinfo('uint32').max)
    random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)
    return seed


def rank_space(data):
    from scipy.stats.mstats import rankdata
    return rankdata(data, axis=0)


def iterate_powerset(items):
    for sub_set in itertools.chain.from_iterable(
            itertools.combinations(items, num) for num in range(1, len(items) + 1)):
        yield sub_set


def named_variations(name):

    # noinspection PyPep8Naming
    load_from_I = LoadFrom('hp_fmri_20', ('hp_fmri_I',), map_run=lambda r: r % 4)

    name = SwitchRemember(name)
    auxiliary_loss_tasks = set()

    if name == 'hp_fmri_meg_joint':
        fmri_subjects_ = ['F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
        # fmri_subjects_ = ['H', 'I', 'K', 'L']
        fmri_tasks_ = tuple('hp_fmri_{}'.format(s) for s in fmri_subjects_)
        training_variations = [
            fmri_tasks_ + ('hp_meg',)]
        # ('hp_meg',),
        # fmri_tasks_]
        settings = Settings(
            corpora=(CorpusTypes.HarryPotterCorpus(
                fmri_subjects=fmri_subjects_,
                fmri_sentence_mode='ignore',
                fmri_window_duration=10.1,
                fmri_minimum_duration_required=9.6,
                group_meg_sentences_like_fmri=True,
                meg_kind='leila',
                meg_subjects=None),),  # None means everyone
            optimization_settings=OptimizationSettings(
                num_train_epochs=60,
                num_epochs_train_prediction_heads_only=10,
                num_final_epochs_train_prediction_heads_only=0))
        settings.preprocessors[ResponseKind.hp_meg] = [
            PreprocessDetrend(
                stop_mode='content', metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessStandardize(
                stop_mode='content', metadata_example_group_by='fmri_runs', train_on_all=True, average_axis=None)]
        settings.preprocessors[ResponseKind.hp_fmri] = [
            PreprocessDetrend(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessStandardize(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True)]
        settings.prediction_heads[ResponseKind.hp_fmri] = PredictionHeadSettings(
            ResponseKind.hp_fmri, KeyedLinear, dict(is_sequence='naked_pooled', force_cpu=True))
        settings.prediction_heads[ResponseKind.hp_meg] = PredictionHeadSettings(
            ResponseKind.hp_meg, head_type=KeyedLinear, kwargs=dict(is_sequence=True, force_cpu=True))
        num_runs = 4
        min_memory = 4 * 1024 ** 3
    elif name == 'hp_meg_simple_fmri_linear':
        fmri_subjects_ = ['F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
        # fmri_subjects_ = ['H', 'I', 'K', 'L']
        training_variations = list()
        for subject in fmri_subjects_:
            training_variations.append(TrainingVariation(
                ('hp_fmri_{}'.format(subject),), load_from=LoadFrom(
                    'hp_fmri_meg',
                    loss_tasks=('hp_meg',))))
            training_variations.append(('hp_fmri_{}'.format(subject),))
        settings = Settings(
            corpora=(CorpusTypes.HarryPotterCorpus(
                fmri_subjects=fmri_subjects_,
                fmri_sentence_mode='ignore',
                fmri_window_duration=10.1,
                fmri_minimum_duration_required=9.6,
                group_meg_sentences_like_fmri=True,
                meg_kind='leila',
                meg_subjects=[]),),  # None means everyone
            optimization_settings=OptimizationSettings(
                num_train_epochs=30,
                num_epochs_train_prediction_heads_only=-1,
                num_final_epochs_train_prediction_heads_only=0),
            filter_when_not_in_loss