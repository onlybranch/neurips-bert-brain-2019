
from dataclasses import dataclass, field
from typing import Sequence, Callable, MutableMapping, Mapping, Optional, Union, Tuple

import numpy as np
from .data_sets import PreprocessStandardize, PreprocessDetrend, HarryPotterMakeLeaveOutFmriRun, PreparedDataView, \
    ResponseKind, InputFeatures, RawData, KindData, CorpusKeys, CorpusBase, HarryPotterCorpus
from .modeling import CriticKeys, FMRIConvConvWithDilationHead


__all__ = ['OptimizationSettings', 'PredictionHeadSettings', 'CriticSettings', 'TrainingVariation', 'LoadFrom',
           'Settings']


@dataclass
class OptimizationSettings:
    # Total number of training epochs to perform.
    num_train_epochs: int = 3
    # initial learning rate for Adam
    learning_rate: float = 5e-5
    # Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% of training.
    warmup_proportion: float = 0.1
    train_batch_size: int = 32
    predict_batch_size: int = 8
    # When splitting up a long document into chunks, how much stride to take between chunks.
    doc_stride: int = 128
    # Whether to perform optimization and keep the optimizer averages on CPU
    optimize_on_cpu: bool = False
    # Whether to use 16-bit float precision instead of 32-bit
    fp16: bool = False
    # Loss scaling, positive power of 2 values can improve fp16 convergence.
    loss_scale: float = 128
    # Number of updates steps to accumulate before performing a backward/update pass.
    gradient_accumulation_steps: int = 1
    # local_rank for distributed training on gpus; probably don't need this
    local_rank: int = -1
    # During the first num_epochs_train_prediction_heads_only, only the prediction heads will be trained
    num_epochs_train_prediction_heads_only: int = 0
    # During the last num_final_prediction_head_only_epochs, only the prediction heads will be trained
    num_final_epochs_train_prediction_heads_only: int = 0


@dataclass
class PredictionHeadSettings:
    key: str
    head_type: type
    kwargs: dict


@dataclass
class CriticSettings:
    critic_type: str
    critic_kwargs: Optional[Mapping] = None


def _default_split_functions():

    return {
        CorpusKeys.HarryPotterCorpus: HarryPotterMakeLeaveOutFmriRun(),
    }


def _default_preprocessors():

    return {
        ResponseKind.hp_fmri: [
            PreprocessDetrend(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessStandardize(stop_mode=None)],
        ResponseKind.hp_meg: [
            PreprocessDetrend(stop_mode='content', metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessStandardize(stop_mode='content', average_axis=None)],
    }


def _default_supplemental_fields():
    return {'token_lengths', 'token_probabilities'}


def _default_prediction_heads():

    return {
        ResponseKind.hp_fmri: PredictionHeadSettings(
            ResponseKind.hp_fmri, FMRIConvConvWithDilationHead, dict(
                hidden_channels=10,
                hidden_kernel_size=5,
                out_kernel_size=5,
                out_dilation=5,
                memory_efficient=False)),
    }


def _default_critics():

    return {
        ResponseKind.hp_fmri: CriticSettings(critic_type=CriticKeys.single_mse),
        CorpusKeys.HarryPotterCorpus: CriticSettings(critic_type=CriticKeys.mse),
    }


@dataclass
class LoadFrom:
    variation_name: str
    loss_tasks: Union[Sequence[str], 'TrainingVariation']
    map_run: Optional[Callable[[int], int]] = None
    name: Optional[str] = None

    def __post_init__(self):
        if self.name is None:
            self.name = '{}:{}'.format(
                self.variation_name,
                self.loss_tasks.name if isinstance(self.loss_tasks, TrainingVariation) else tuple(self.loss_tasks))


@dataclass
class TrainingVariation:
    loss_tasks: Sequence[str]