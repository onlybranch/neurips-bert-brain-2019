import inspect
from itertools import zip_longest
import re


__all__ = ['zip_equal', 'copy_from_properties', 'get_keyword_properties', 'SwitchRemember', 'camel_to_snake',
           'MultiReplace', 'split_with_indices']


def zip_equal(*it):
    """
    Like zip, but raises a ValueError if the iterables are not of equal length
    Args:
        *it: The iterables to zip

    Returns:
        yields a tuple of items, one from each iterable
    """
    # wrap the iterators in an enumerate to guarantee that None is a legitimate sentinel
    iterators = [enumerate(i) for i in it]
    for idx, item in enumerate(zip_longest(*ite