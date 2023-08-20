
import os
from collections import OrderedDict
import dataclasses

import numpy as np
import torch

from .input_features import InputFeatures, RawData, KindData, FieldSpec


__all__ = ['save_to_cache', 'load_from_cache']


def save_to_cache(cache_path, data, run_info, kwargs):

    cache_dir, _ = os.path.split(cache_path)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    num_input_examples = 0 if data.input_examples is None else len(data.input_examples)
    has_input_examples = data.input_examples is not None