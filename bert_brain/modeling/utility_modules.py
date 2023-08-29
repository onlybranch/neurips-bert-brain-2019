import numpy as np
import torch


__all__ = ['GroupPool', 'GroupConcat', 'Conv1DCausal', 'at_most_one_data_id', 'k_data_ids']


def at_most_one_data_id(data_ids, return_first_index=False, return_last_index=False):

    if len(data_ids.size()) != 2:
        raise ValueError('data_ids must be 2D')

    maxes, _ = torch.max(data_ids, dim=1)
    repeated_maxes = torch.reshape(maxes, (-1, 1)).repeat((1, data_ids.size()[1]))
    mins, _ = torch.min(torch.where(data_ids < 0, repeated_maxes, data_ids), dim=1)

    if torch.sum(maxes != mins) > 0:
        raise ValueError('More than one data_id exists for some examples')

    if return_first_index or return_last_index:
        index_array = torch.arange(data_ids.size()[1], device=data_ids.device).view(
            (1, data_ids.size()[1])).repeat((data_ids.size()[0], 1))
        indicator_valid = data_ids >= 0
        first_index = None
        if return_first_index:
            first_index_array = torch.where(
                indicator_valid, index_array, torch.full_like(index_array, data_ids.size()[1] + 1))
            first_index, _ = torch.min(first_index_array, dim=1)
        last_index = None
        if return_last_index:
            last_index_array = torch.where(indicator_valid, index_array, torch.full_like(index_array, -1))
            last_index, _ = torch.max(last_index_array, dim=1)
        if return_first_index and return_last_index:
            return maxes, first_index, last_index
        if return_first_index:
            return maxes, first_index
        if return_last_index:
            return maxes, last_index

    return maxes


def k_data_ids(k, data_ids, return_indices=False, check_unique=False):

    if len(data_ids.size()) != 2:
        raise ValueError('data_ids must be 2D')

    indicator_valid = data_ids >= 0
    count_valid = torch.sum(indicat