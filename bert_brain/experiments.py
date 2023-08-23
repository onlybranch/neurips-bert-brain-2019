import hashlib
import itertools
import random

import numpy as np
import torch

from .common import SwitchRemember
from .data_sets import ResponseKind, CorpusTypes, PreprocessDetrend, PreprocessStandardize
from .modeling import KeyedLinear
from .settings import TrainingV