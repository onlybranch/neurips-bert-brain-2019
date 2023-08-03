import warnings
from collections import OrderedDict

import numpy as np

from bert_brain import TrainingVariation, read_variation_results
from ocular import TextGrid, TextWrapStyle, write_text_grid_to_console


output_order = (
    'mse',       # mean squared error
    'mae',       # mean absolute error
    'pove',      # proportion of variance explained
    'povu',      # proportion of variance unexplained
    'podu',      # proportion of mean absolute deviation unexplained
    'pode',      # proportion of mean absolute deviation explained
    'variance',
    'mad',       # mean absolute deviation
    'r_seq',     # avg (over batch) of sequence correlation values (i.e. correlation within a sequence)
    'xent',      # cross entropy
    'acc',       # accuracy
    'macc',      # mode accuracy - the accuracy one would get if one picked the mode
    'poma',      # proportion of mode accuracy; < 1 is bad
    'prec',      # precision
    'rec',       # recall
    'f1')


def print_variation_results_sliced