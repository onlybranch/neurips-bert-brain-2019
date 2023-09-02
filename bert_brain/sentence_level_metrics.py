import os
from .experiments import named_variations, match_variation, task_hash
from .result_output import read_predictions


__all__ = ['sentence_predictions']


def sentence_predictions(paths, variation_set_name, training_variation, key):
    _, _, num_runs, _, _ = named_variations(variation_set_name)
    training_variation = match_variation(var