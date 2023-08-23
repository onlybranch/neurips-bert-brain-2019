
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from itertools import chain
from dataclasses import replace, asdict
from typing import Optional
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import sosfilt
from sklearn.decomposition import PCA

from .input_features import InputFeatures, KindData

__all__ = [
    'PreprocessLog',
    'PreprocessDetrend',
    'PreprocessDiscretize',
    'PreprocessBaseline',
    'PreprocessFeatureStandardize',
    'PreprocessFeatureNormalize',
    'PreprocessSequenceStandardize',
    'PreprocessStandardize',
    'PreprocessMakeBinary',
    'PreprocessNanMean',
    'PreprocessPCA',
    'PreprocessNanGeometricMean',
    'PreprocessClip',
    'PreprocessGaussianBlur',
    'PreprocessCompress',
    'PreprocessSqueeze']


def _indicator_from_examples(data_size, examples, stop_mode=None):
    result = np.full(data_size, False)
    for ex in examples:
        data_ids = ex.data_ids
        if stop_mode is not None:
            if stop_mode == 'content':
                data_ids = np.where(ex.is_stop, -1, data_ids)
            elif stop_mode == 'stop':
                data_ids = np.where(ex.is_stop, data_ids, -1)
            else:
                raise ValueError('Unknown value for stop_mode: {}'.format(stop_mode))
        data_ids = data_ids[data_ids >= 0]
        result[data_ids] = True
    return result


def _unsorted_group_by(items, group_by_fn):
    groups = dict()
    for item in items:
        key = group_by_fn(item)
        if key not in groups:
            groups[key] = list()
        groups[key].append(item)
    for key in sorted(groups):
        yield key, groups[key]


def _parallel_column_map(fit_fn, apply_fn, data, indicator_fit=None):
    shape = data.shape
    data = np.reshape(data, (data.shape[0], -1))
    with ProcessPoolExecutor() as ex:
        fit_result = list(ex.map(fit_fn, [(data[:, i], indicator_fit) for i in range(data.shape[1])]))
    assert(len(fit_result) == data.shape[1])
    data = apply_fn(data, fit_result)
    return np.reshape(data, shape)


class PreprocessLog:

    def __init__(self, min_value: float = -20.):
        self.min_value = min_value

    def __call__(self, loaded_data_tuple, metadata, random_state):
        isnan = np.isnan(loaded_data_tuple.data)
        data = np.where(isnan, 1, loaded_data_tuple.data)
        if np.any(np.less(data, 0)):
            raise ValueError('Values must be >= 0')
        data = np.log(np.maximum(data, np.power(np.e, self.min_value)))
        return replace(loaded_data_tuple, data=np.where(isnan, np.nan, data))


def _lin_regress(item):
    from scipy.stats import linregress
    y, indicator_train = item
    x = np.arange(len(y))

    if indicator_train is not None:
        x = x[indicator_train]
        y = y[indicator_train]

    indicator_valid = np.logical_not(np.isnan(y))
    x = x[indicator_valid]
    y = y[indicator_valid]

    if len(x) == 0:
        return 0, 0

    m, b, _, _, _ = linregress(x, y)
    return m, b


def _remove_lin_regress(data, p):
    p = np.concatenate(list(np.expand_dims(p_col, 1) for p_col in p), axis=1)
    #      (1, num_columns)            (num_rows, 1)
    lines = np.reshape(p[0], (1, -1)) * np.reshape(np.arange(len(data)), (-1, 1)) + np.reshape(p[1], (1, -1))
    lines = np.reshape(lines, data.shape)
    return data - lines


class PreprocessDetrend:

    def __init__(
            self,
            stop_mode: Optional[str] = None,
            metadata_example_group_by: str = None,
            train_on_all: bool = False):
        self.stop_mode = stop_mode
        self.metadata_example_group_by = metadata_example_group_by
        self.train_on_all = train_on_all

    @staticmethod
    def _detrend(arr, indicator_train):
        return _parallel_column_map(_lin_regress, _remove_lin_regress, arr, indicator_train)

    def __call__(self, loaded_data_tuple, metadata, random_state):
        train_examples = loaded_data_tuple.train
        if self.train_on_all:
            train_examples = chain(loaded_data_tuple.train, loaded_data_tuple.validation, loaded_data_tuple.test)
        indicator_train = _indicator_from_examples(
            len(loaded_data_tuple.data), train_examples, self.stop_mode)

        if self.metadata_example_group_by is not None:
            if metadata is None or self.metadata_example_group_by not in metadata:
                raise ValueError('metadata_example_group_by {} not found in metadata'.format(
                    self.metadata_example_group_by))
            data = np.copy(loaded_data_tuple.data)
            grouped_examples = _unsorted_group_by(
                chain(loaded_data_tuple.train, loaded_data_tuple.validation, loaded_data_tuple.test),
                lambda ex: metadata[self.metadata_example_group_by][ex.unique_id])
            for group, group_examples in grouped_examples:
                indicator_group = _indicator_from_examples(len(data), group_examples)
                group_data = data[indicator_group]
                group_indicator_train = indicator_train[indicator_group] if indicator_train is not None else None
                group_data = PreprocessDetrend._detrend(group_data, group_indicator_train)
                data[indicator_group] = group_data
        else:
            data = PreprocessDetrend._detrend(loaded_data_tuple.data, indicator_train)

        return replace(loaded_data_tuple, data=data)


class PreprocessDiscretize:

    # noinspection PyShadowingBuiltins
    def __init__(self, bins=10, range=None, use_one_hot=True):
        self.bins = bins
        self.range = range
        self.use_one_hot = use_one_hot

    # noinspection PyShadowingBuiltins
    def __call__(self, loaded_data_tuple, metadata, random_state):
        bin_edges = np.histogram_bin_edges(loaded_data_tuple.data, self.bins, range)
        if np.isscalar(self.bins):
            bin_edges = bin_edges[1:]
        data = np.digitize(loaded_data_tuple.data, bin_edges, right=True)
        if self.use_one_hot:
            one_hot = np.zeros(data.shape + (len(bin_edges) + 1,), data.dtype)
            one_hot = np.reshape(one_hot, (-1, one_hot.shape[-1]))
            for idx, bin in enumerate(np.reshape(data, -1)):
                one_hot[idx, bin] = 1
            data = np.reshape(one_hot, data.shape + (one_hot.shape[-1],))
        return replace(loaded_data_tuple, data=data)


class PreprocessBaseline:

    def __init__(self, num_baseline):
        """
        Computes a running mean using a window of num_baseline values and subtracts this running mean
        from the data. This completely ignores example boundaries. Validation/test examples are removed if the baselines
        from those examples would overlap with train examples
        """
        self.num_baseline = num_baseline

    def _find_max_mins(self, examples, keep_if_greater_than=None):
        if examples is None:
            return None, None, examples
        data_ids = np.concatenate([ex.data_ids for ex in examples])
        data_ids = data_ids[data_ids >= 0]
        max_id = np.max(data_ids)
        min_id = np.min(data_ids)
        if keep_if_greater_than is None:
            return max_id, min_id, examples
        clean = list()
        for ex in examples:
            ex_min = np.min(ex.data_idx[ex.data_ids >= 0])
            if ex_min - self.num_baseline > keep_if_greater_than:
                clean.append(ex)
        return max_id, min_id, clean

    def _compute_baseline(self, arr):
        indicator_nan = np.isnan(arr)
        result = np.cumsum(np.where(indicator_nan, 0, arr), axis=0)
        if len(result) > self.num_baseline:
            result[self.num_baseline:] = result[self.num_baseline:] - result[:-self.num_baseline]
        counts = np.cumsum(np.logical_not(indicator_nan))
        if len(counts) > self.num_baseline:
            counts[self.num_baseline:] = counts[self.num_baseline:] - counts[:-self.num_baseline]
        return result / counts

    def _subtract_baseline(self, examples, data):
        if examples is None:
            return
        data_ids = np.concatenate([ex.data_ids for ex in examples])
        data_ids = data_ids[data_ids >= 0]
        baseline = self._compute_baseline(data[data_ids])
        data[data_ids] = data[data_ids] - baseline

    def __call__(self, loaded_data_tuple, metadata, random_state):
        max_train, _, _ = self._find_max_mins(loaded_data_tuple.train)
        _, _, clean_validation = self._find_max_mins(loaded_data_tuple.validation, keep_if_greater_than=max_train)
        loaded_data_tuple = replace(loaded_data_tuple, validation=clean_validation)
        _, _, clean_test = self._find_max_mins(loaded_data_tuple.test, keep_if_greater_than=max_train)
        loaded_data_tuple = replace(loaded_data_tuple, test=clean_test)

        data = np.copy(loaded_data_tuple.data)

        self._subtract_baseline(loaded_data_tuple.train, data)
        self._subtract_baseline(loaded_data_tuple.validation, data)
        self._subtract_baseline(loaded_data_tuple.test, data)

        return replace(loaded_data_tuple, data=data)


class PreprocessFeatureStandardize:

    def __init__(self):
        pass

    def __call__(self, loaded_data_tuple, metadata, random_state):
        d = np.reshape(loaded_data_tuple.data, (loaded_data_tuple.data.shape[0], -1))
        d = (d - np.nanmean(d, axis=1, keepdims=True)) / np.nanstd(d, axis=1, keepdims=True)
        return replace(loaded_data_tuple, data=np.reshape(d, loaded_data_tuple.data.shape))


class PreprocessFeatureNormalize:

    def __init__(self):
        pass

    def __call__(self, loaded_data_tuple, metadata, random_state):
        d = np.reshape(loaded_data_tuple.data, (loaded_data_tuple.data.shape[0], -1))
        n = np.nansum(np.abs(d), axis=1, keepdims=True)
        d = np.divide(d, n, where=n != 0)
        return replace(loaded_data_tuple, data=np.reshape(d, loaded_data_tuple.data.shape))


class PreprocessSequenceStandardize:

    def __init__(self, stop_mode):
        self.stop_mode = stop_mode

    def __call__(self, loaded_data_tuple, metadata, random_state):

        data = np.copy(loaded_data_tuple.data)
        for ex in chain(loaded_data_tuple.train, loaded_data_tuple.validation, loaded_data_tuple.test):

            data_indices = ex.data_ids
            compute_indices = data_indices
            if self.stop_mode == 'content':
                compute_indices = np.where(ex.is_stop, -1, compute_indices)
            elif self.stop_mode == 'stop':