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


def print_variation_results_sliced(
        paths, variation_set_name, training_variation, aux_loss, num_runs, metric='pove',
        field_precision=2, num_values_per_table=10, **loss_handler_kwargs):

    aggregated, count_runs = read_variation_results(paths, variation_set_name, training_variation, aux_loss, num_runs,
                                                    compute_scalar=False, **loss_handler_kwargs)

    values = OrderedDict((name, np.nanmean(aggregated[name].values(metric), axis=0)) for name in aggregated)

    grouped_by_shape = OrderedDict()
    for name in values:
        if values[name].shape not in grouped_by_shape:
            grouped_by_shape[values[name].shape] = [name]
        else:
            grouped_by_shape[values[name].shape].append(name)

    print('Variation ({} of {} runs found): {}'.format(count_runs, num_runs, ', '.join(sorted(training_variation))))

    for shape in grouped_by_shape:
        num_tables = int(np.ceil(np.prod(shape) / num_v