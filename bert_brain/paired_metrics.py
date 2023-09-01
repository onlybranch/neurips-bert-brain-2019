
import warnings
import numpy as np
from scipy.stats import ttest_1samp, ttest_rel

from tqdm import trange

from .experiments import named_variations, match_variation
from .aggregate_metrics import get_field_predictions


__all__ = [
    'paired_squared_error',
    'sorted_cumulative_mean_diff',
    'PermutationTestResult',
    'sample_differences',
    'one_sample_permutation_test',
    'two_sample_permutation_test',
    'get_k_vs_k_paired',
    'get_mse_paired',
    'wilcoxon_axis',
    'ResultPValues']


def paired_squared_error(
        paths_obj,
        variation_set_name_a, training_variation_a,
        variation_set_name_b, training_variation_b,
        field_name,
        num_contiguous):
    predictions_a, target_a, ids_a = get_field_predictions(
        paths_obj, variation_set_name_a, training_variation_a, field_name)
    predictions_b, target_b, ids_b = get_field_predictions(
        paths_obj, variation_set_name_b, training_variation_b, field_name)

    def _sort(p, t, i):
        sort_order = np.argsort(i)
        return p[sort_order], t[sort_order], i[sort_order]

    predictions_a, target_a, ids_a = _sort(predictions_a, target_a, ids_a)
    predictions_b, target_b, ids_b = _sort(predictions_b, target_b, ids_b)

    err_a = np.square(predictions_a - target_a)
    err_b = np.square(predictions_b - target_b)

    if not np.array_equal(ids_a, ids_b):
        raise ValueError('Mismatched example ids')

    def _mean_contiguous(e):
        return np.array(
            list(np.nanmean(item, axis=0) for item in np.array_split(e, int(np.ceil(len(e) / num_contiguous)))))

    mse_a = np.nanmean(err_a, axis=0)
    mse_b = np.nanmean(err_b, axis=0)
    pove_a = 1 - mse_a / np.nanvar(target_a, axis=0)
    pove_b = 1 - mse_b / np.nanvar(target_b, axis=0)
    err_a = _mean_contiguous(err_a)
    err_b = _mean_contiguous(err_b)
    return err_a - err_b, pove_a, pove_b


def _k_vs_k_accuracy(predictions, target, indices_true, indices_distractor):
    sample_target = target[indices_true]
    sample_distractor = predictions[indices_distractor]
    sample_predictions = predictions[indices_true]

    sample_target = np.reshape(sample_target, (-1, sample_target.shape[-1]))
    sample_distractor = np.reshape(sample_distractor, (-1, sample_distractor.shape[-1]))
    sample_predictions = np.reshape(sample_predictions, (-1, sample_predictions.shape[-1]))

    distance_correct = np.sum((sample_target - sample_predictions) ** 2, axis=0)
    distance_incorrect = np.sum((sample_target - sample_distractor) ** 2, axis=0)
    return (distance_incorrect > distance_correct) * 1.0 + (distance_incorrect == distance_correct) * 0.5


class PermutationTestResult:
    def __init__(self, true_values, permutation_values, p_values):
        self.true_values = true_values
        self.permutation_values = permutation_values
        self.p_values = p_values


def paired_k_vs_k_permutation(
        paths_obj,
        variation_set_name_a, training_variation_a, variation_set_name_b, training_variation_b, field_name,
        k=20, num_k_vs_k_samples=100, num_permutations=1000):
    _, _, num_runs_a, _, _ = named_variations(variation_set_name_a)
    _, _, num_runs_b, _, _ = named_variations(variation_set_name_b)
    assert (num_runs_a == num_runs_b)
    training_variation_a = match_variation(variation_set_name_a, training_variation_a)
    training_variation_b = match_variation(variation_set_name_b, training_variation_b)

    results_a = list()
    results_b = list()
    results_sum = list()
    results_diff = list()

    def block_permute_indices(count, block_size):
        permute_indices_ = np.random.permutation(int(np.ceil(count / block_size)))
        permute_indices_ = np.reshape(
            permute_indices_ * block_size, (-1, 1)) + np.reshape(np.arange(block_size), (1, -1))
        permute_indices_ = np.reshape(permute_indices_, -1)
        return permute_indices_[permute_indices_ < count]

    def read_results(variation_name, training_variation, idx_run):
        p, t, ids = get_field_predictions(
            paths_obj, variation_name, training_variation, field_name, idx_run, pre_matched=True)
        sort_order = np.argsort(ids)
        return p[sort_order], t[sort_order]

    for index_run in range(num_runs_a):
        predictions_a, target_a = read_results(variation_set_name_a, training_variation_a, index_run)
        predictions_b, target_b = read_results(variation_set_name_b, training_variation_b, index_run)
        assert (len(target_a) == len(target_b))

        for index_permutation in trange(num_permutations + 1):

            if index_permutation == 0:
                permuted_target_a = target_a
                permuted_target_b = target_b
            else:
                # permute in blocks of 10
                permute_indices = block_permute_indices(len(target_a), block_size=10)
                permuted_target_a = target_a[permute_indices]
                permuted_target_b = target_b[permute_indices]

            accuracy_a = None
            accuracy_b = None
            for index_k_vs_k_sample in range(num_k_vs_k_samples):
                if index_k_vs_k_sample == 0:
                    accuracy_a = np.full((num_k_vs_k_samples, target_a.shape[-1]), np.nan)
                    accuracy_b = np.full((num_k_vs_k_samples, target_b.shape[-1]), np.nan)
                indices_true = np.random.choice(len(target_a), k)
                indices_distractor = np.random.choice(len(target_a), k)
                accuracy_a[index_k_vs_k_sample] = _k_vs_k_accuracy(
                    predictions_a, permuted_target_a, indices_true, indices_distractor)
                accuracy_b[index_k_vs_k_sample] = _k_vs_k_accuracy(
                    predictions_b, permuted_target_b, indices_true, indices_distractor)

            mean_a = np.mean(accuracy_a, axis=0)
            mean_b = np.mean(accuracy_b, axis=0)

            for vals, results in [
                    (mean_a, results_a),
                    (mean_b, results_b),
                    (mean_a + mean_b, results_sum),
                    (mean_a - mean_b, results_diff)]:
                if index_permutation == 0:
                    results.append(
                        PermutationTestResult(
                            vals,