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
            if self.