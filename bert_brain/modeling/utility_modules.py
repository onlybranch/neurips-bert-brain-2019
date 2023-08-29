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
    count_valid = torch.sum(indicator_valid, dim=1)
    if torch.max(count_valid) != k or torch.min(count_valid) != k:
        print(count_valid)
        raise ValueError('Incorrect number of data_ids. Expected {}'.format(k))

    data_ids = torch.masked_select(data_ids, indicator_valid)
    data_ids = torch.reshape(data_ids, (data_ids.size()[0], k))

    if check_unique:
        mins, _ = torch.min(data_ids, dim=1)
        maxes, _ = torch.max(data_ids, dim=1)
        if torch.sum(maxes != mins) > 0:
            raise ValueError('More than one data_id exists for some examples')

    if return_indices:
        index_array = torch.arange(data_ids.size()[1], device=data_ids.device).view(
            (1, data_ids.size()[1])).repeat((data_ids.size()[0], 1))
        indices = torch.masked_select(index_array, indicator_valid)
        indices = torch.reshape(indices, (indicator_valid.size()[0], k))
        return data_ids, indices

    return data_ids


class GroupConcat(torch.nn.Module):

    def __init__(self, num_per_group):
        super().__init__()
        self.num_per_group = num_per_group

    # noinspection PyMethodMayBeStatic
    def forward(self, x, groupby):

        # first attach an example_id to the groups to ensure that we don't concat across examples in the batch

        # array of shape (batch, sequence, 1) which identifies example
        example_ids = torch.arange(
            groupby.size()[0], device=x.device).view((groupby.size()[0], 1, 1)).repeat((1, groupby.size()[1], 1))

        # indices to ensure stable sort, and to give us indices_sort
        indices = torch.arange(groupby.size()[0] * groupby.size()[1], device=x.device).view(groupby.size() + (1,))

        # -> (batch, sequence, 3): attach example_id to each group and add indices to guarantee stable sort
        groupby = torch.cat((example_ids, groupby.view(groupby.size() + (1,)), indices), dim=2)

        # -> (batch * sequence, 3)
        groupby = groupby.view((g