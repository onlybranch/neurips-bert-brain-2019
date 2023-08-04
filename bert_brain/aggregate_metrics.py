
import fnmatch
import inspect
import os
import warnings
from collections import OrderedDict
import dataclasses
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
from typing import Optional

import numpy as np
from scipy.special import logsumexp

from .experiments import named_variations, task_hash, match_variation
from .settings import TrainingVariation
from .modeling import CriticMapping
from .result_output import read_predictions

__all__ = [
    'Aggregator',
    'read_variation_results',
    'nan_pearson',
    'regression_handler',
    'class_handler',
    'bincount_axis',
    'make_prediction_handler',
    'ResultQuery',
    'query_results',
    'get_field_predictions',
    'k_vs_k']


class Aggregator:
    def __init__(self):
        """
        Helper class to aggregate metrics over runs etc.
        """
        self._field_values = None
        self._counts = None

    def update(self, result, is_sequence):
        if self._field_values is None:
            self._field_values = OrderedDict()
            self._counts = OrderedDict()
            if dataclasses.is_dataclass(result):
                for field in dataclasses.fields(result):
                    self._field_values[field.name] = list()
                    self._counts[field.name] = list()
            else:
                for field in result:
                    self._field_values[field] = list()
                    self._counts[field] = list()

        if dataclasses.is_dataclass(result):
            result = dataclasses.asdict(result)
        for field in result:
            if field not in self._field_values:
                raise ValueError('Unexpected field in result: {}'.format(field))
            if result[field] is None:
                self._counts[field].append(0)
            elif np.isscalar(result[field]):
                self._field_values[field].append(result[field])
                self._counts[field].append(1)
            elif is_sequence:
                self._field_values[field].extend(result[field])
                self._counts[field].append(len(result[field]))
            else:
                self._field_values[field].append(result[field])
                self._counts[field].append(1)

    def __contains__(self, item):
        return item in self._field_values

    def __iter__(self):
        for k in self._field_values:
            yield k

    def __getitem__(self, item):
        return self._field_values[item]

    def value_dict(self, names=None, fn=None, value_on_key_error=None):
        if names is None:
            if fn is None:
                return OrderedDict(self._field_values)
            return OrderedDict((k, fn(self._field_values[k])) for k in self._field_values)
        if isinstance(names, str):
            names = [names]
        result = OrderedDict()
        for name in names:
            if value_on_key_error is not None and name not in self._field_values:
                result[name] = value_on_key_error
            else:
                result[name] = fn(self._field_values[name]) if fn is not None else self._field_values[name]
        return result

    def values(self, name, fn=None):
        if fn is None:
            return self._field_values[name]
        return fn(self._field_values[name])

    def counts(self, name):
        return self._counts[name]


def read_no_cluster_data(path):
    with np.load(path) as loaded:
        unique_ids = loaded['unique_ids']
        lengths = loaded['lengths']
        data_ids = loaded['data_ids']
        splits = np.cumsum(lengths)[:-1]
        data_ids = np.split(data_ids, splits)
        return unique_ids, data_ids, loaded['data']


def expand_predictions(prediction, cluster_ids):
    is_prediction_1d = len(prediction.shape) == 1
    if is_prediction_1d:
        prediction = np.expand_dims(prediction, 0)
    expanded = np.zeros((prediction.shape[0], np.prod(cluster_ids.shape)), prediction.dtype)
    for idx, c in enumerate(np.unique(cluster_ids)):
        indicator = cluster_ids == c
        expanded[:, indicator] = prediction[:, idx]
    if is_prediction_1d:
        return np.reshape(expanded, cluster_ids.shape)
    else:
        return np.reshape(expanded, (prediction.shape[0],) + cluster_ids.shape)


def _read_variation_parallel_helper(item):
    (result_path, model_path, variation_set_name, variation_hash, index_run, aux_loss,
     compute_scalar, k_vs_k_feature_axes, loss_handler_kwargs) = item
    training_variations, _, _, _, _ = named_variations(variation_set_name)
    training_variation = None
    for v in training_variations:
        if task_hash(v) == variation_hash:
            training_variation = v
            break
    if training_variation is None:
        raise RuntimeError('Bad variation hash')
    output_dir = os.path.join(result_path, variation_set_name, variation_hash)
    model_dir = os.path.join(model_path, variation_set_name, variation_hash, 'run_{}'.format(index_run))
    validation_npz_path = os.path.join(output_dir, 'run_{}'.format(index_run), 'output_validation.npz')
    if not os.path.exists(validation_npz_path):
        return index_run, None
    output_results_by_name = read_predictions(validation_npz_path)
    run_results = dict()
    for name in output_results_by_name:
        if isinstance(training_variation, TrainingVariation):
            in_training_variation = name in training_variation.loss_tasks
        else:
            in_training_variation = name in training_variation
        if not in_training_variation and name not in aux_loss:
            continue
        no_cluster_path = os.path.join(model_dir, '{}_no_cluster_to_disk.npz'.format(name))
        cluster_id_path = os.path.join(model_dir, 'kmeans_clusters_{}.npy'.format(name))
        cluster_ids = None
        no_cluster_unique_ids = None
        no_cluster_data_ids = None
        no_cluster_data = None
        if os.path.exists(cluster_id_path) and os.path.exists(no_cluster_path):
            cluster_ids = np.load(cluster_id_path)
            no_cluster_unique_ids, no_cluster_data_ids, no_cluster_data = read_no_cluster_data(no_cluster_path)
        output_results = output_results_by_name[name]
        run_aggregated = Aggregator()
        loss = None
        for output_result in output_results:
            if loss is None:
                loss = output_result.critic_type
            else:
                assert (loss == output_result.critic_type)
            if cluster_ids is not None:
                output_result.prediction = expand_predictions(output_result.prediction, cluster_ids)
                output_result.mask = expand_predictions(output_result.mask, cluster_ids)
                index_unique_id = np.where(output_result.unique_id == no_cluster_unique_ids)[0]
                assert(len(index_unique_id) == 1)
                index_unique_id = index_unique_id[0]
                data_ids = no_cluster_data_ids[index_unique_id]
                data_ids = data_ids[data_ids >= 0]
                seen = set()
                unique_data_ids = list()
                for d in data_ids:
                    if d not in seen:
                        unique_data_ids.append(d)
                        seen.add(d)
                assert(len(unique_data_ids) == output_result.target.shape[0])
                output_result.target = np.array(list([no_cluster_data[d] for d in unique_data_ids]))
            run_aggregated.update(output_result, is_sequence=output_result.sequence_type != 'single')

        loss_handler_kwargs = dict(loss_handler_kwargs)
        if isinstance(k_vs_k_feature_axes, dict):
            if name in k_vs_k_feature_axes:
                loss_handler_kwargs['k_vs_k_feature_axes'] = k_vs_k_feature_axes[name]
            else:
                loss_handler_kwargs['k_vs_k_feature_axes'] = -1
        else:
            loss_handler_kwargs['k_vs_k_feature_axes'] = k_vs_k_feature_axes
        handler = make_prediction_handler(loss, loss_handler_kwargs)
        result_dict = handler(run_aggregated)
        if compute_scalar:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                result_dict = dict((k, np.nanmean(result_dict[k])) for k in result_dict)
        run_results[name] = result_dict
    return index_run, run_results


def read_variation_results(paths, variation_set_name, training_variation, aux_loss, num_runs,
                           compute_scalar=True, k_vs_k_feature_axes=-1, **loss_handler_kwargs):

    task_arguments = [(paths.result_path, paths.model_path, variation_set_name, task_hash(training_variation), i,
                       aux_loss, compute_scalar, k_vs_k_feature_axes, loss_handler_kwargs) for i in range(num_runs)]

    with ThreadPoolExecutor() as ex:
        mapped = ex.map(_read_variation_parallel_helper, task_arguments)
    # mapped = map(_read_variation_parallel_helper, task_arguments)

    has_warned = False
    count_runs = 0
    aggregated = dict()
    for index_run, run_results in mapped:
        if run_results is None:
            if not has_warned:
                print('Warning: results incomplete. Some output files not found')
            has_warned = True
            continue

        count_runs += 1
        for name in run_results:
            if name not in aggregated:
                aggregated[name] = Aggregator()
            aggregated[name].update(run_results[name], is_sequence=False)

    return aggregated, count_runs


def nan_pearson(x, y, axis=0, keepdims=False):
    if not np.array_equal(x.shape, y.shape):
        raise ValueError('x and y must be the same shape')
    if np.isscalar(x):
        raise ValueError('x and y must not be scalar')
    if np.prod(x.shape) == 0:
        result = np.full_like(x, np.nan)
        if x.shape[axis] < 1:
            print(x.shape)
            raise ValueError('x and y must have at least 2 values')
        result = np.take(result, [0], axis=axis)
        if not keepdims:
            result = np.squeeze(result, axis=axis)
        return result