
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from shutil import rmtree
import argparse
import logging
import os
from dataclasses import replace
from typing import Sequence, Union
from tqdm import tqdm
from tqdm_logging import replace_root_logger_handler

import torch

from bert_brain import cuda_most_free_device, cuda_auto_empty_cache_context, DataPreparer, CorpusLoader, \
    Settings, TrainingVariation, task_hash, set_random_seeds, named_variations, train, make_datasets
from bert_brain_paths import Paths


__all__ = ['run_variation']


replace_root_logger_handler()
logger = logging.getLogger(__name__)


def progress_iterate(iterable, progress_bar):
    for item in iterable:
        yield item
        progress_bar.update()


def run_variation(
            set_name,
            loss_tasks: Union[Sequence[str], TrainingVariation],
            settings: Settings,
            num_runs: int,
            auxiliary_loss_tasks: Sequence[str],
            force_cache_miss: bool,
            device: torch.device,
            n_gpu: int,
            progress_bar=None):

    if settings.optimization_settings.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            settings.optimization_settings.gradient_accumulation_steps))

    # TODO: seems like this is taken care of below?
    # settings = replace(
    #   settings, train_batch_size=int(settings.train_batch_size / settings.gradient_accumulation_steps))

    def io_setup():
        hash_ = task_hash(loss_tasks)
        paths_ = Paths()
        paths_.model_path = os.path.join(paths_.model_path, set_name, hash_)
        paths_.result_path = os.path.join(paths_.result_path, set_name, hash_)

        corpus_loader_ = CorpusLoader(paths_.cache_path)

        if not os.path.exists(paths_.model_path):
            os.makedirs(paths_.model_path)
        if not os.path.exists(paths_.result_path):
            os.makedirs(paths_.result_path)

        return corpus_loader_, paths_

    corpus_loader, paths = io_setup()

    load_from = None
    if isinstance(loss_tasks, TrainingVariation):
        load_from = loss_tasks.load_from
        loss_tasks = set(loss_tasks.loss_tasks)
    else:
        loss_tasks = set(loss_tasks)

    loss_tasks.update(auxiliary_loss_tasks)
    settings = replace(settings, loss_tasks=loss_tasks)

    if progress_bar is None:
        progress_bar = tqdm(total=num_runs, desc='Runs')

    for index_run in progress_iterate(range(num_runs), progress_bar):
