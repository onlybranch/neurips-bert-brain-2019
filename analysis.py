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
        num_tables = int(np.ceil(np.prod(shape) / num_values_per_table))
        for i in range(num_tables):
            indices = np.arange(num_values_per_table) + i * num_values_per_table
            indices = indices[indices < np.prod(shape)]
            indices = np.unravel_index(indices, shape)

            text_grid = TextGrid()
            text_grid.append_value('name', column_padding=2)
            # indices is a tuple of arrays, length 1 is a special case
            for index in indices[0] if len(indices) == 1 else zip(indices):
                text_grid.append_value('{}'.format(index), line_style=TextWrapStyle.right_justify, column_padding=2)
            text_grid.next_row()
            value_format = '{' + ':.{}f'.format(field_precision) + '}'
            for name in grouped_by_shape[shape]:
                text_grid.append_value(name, column_padding=2)
                current_values = values[name][indices]
                for value in current_values:
                    text_grid.append_value(
                        value_format.format(value), line_style=TextWrapStyle.right_justify, column_padding=2)
                text_grid.next_row()

            write_text_grid_to_console(text_grid, width='tight')
            print('')

    print('')
    print('')


def print_variation_results(paths, variation_set_name, training_variation, aux_loss, num_runs, field_precision=2,
                            **loss_handler_kwargs):

    aggregated, count_runs = read_variation_results(paths, variation_set_name, training_variation, aux_loss, num_runs,
                                                    **loss_handler_kwargs)

    metrics = list()
    for metric in output_order:
        if any(metric in aggregated[name] for name in aggregated):
            metrics.append(metric)

    text_grid = TextGrid()
    text_grid.append_value('name', column_padding=2)
    for metric in metrics:
        text_grid.append_value(metric, line_style=TextWrapStyle.right_justify, column_padding=2)
    text_grid.next_row()
    value_format = '{' + ':.{}f'.format(field_precision) + '}'
    for name in aggregated:
        text_grid.append_value(name, column_padding=2)
        for metric in metrics:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                value = np.nanmean(aggregated[name].values(metric)) if metric in aggregated[name] else np.nan
            text_grid.append_value(value_format.format(value), line_style=TextWrapStyle.right_justify, column_padding=2)
        text_grid.next_row()

    if isinstance(training_variation, TrainingVariation):
        training_variation_name = str(training_variation)
    else:
        training_variation_name = ', '.join(sorted(training_variation))
    print('Variation ({} of {} runs found): {}'.format(count_runs, num_runs, training_variation_name))
    write_text_grid_to_console(text_grid, width='tight')
    p