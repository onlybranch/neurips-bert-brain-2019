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
   