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
    for idx, item in enumerate(zip_longest(*iterators)):
        try:
            result = tuple(part[1] for part in item)
            yield result
        except TypeError:
            culprit = None
            for idx_part, part in enumerate(item):
                if part is None:
                    culprit = idx_part
                    break
            raise ValueError(
                'Unequal number of elements in iterators. Problem occurred at index: {}, iterator_index: {}'.format(
                    idx, culprit))


def copy_from_properties(instance, **kwargs):
    """
    Returns a copy of instance by calling __init__ with keyword arguments matching the properties of instance.
    The values of these keyword arguments are taken from the properties of instance except where overridden by
    kwargs. Thus for a class Foo with properties [a, b, c], copy_from_properties(instance, a=7) is equivalent to
    Foo(a=7, b=instance.b, c=instance.c)
    Notes:
        Now that Python includes dataclasses, using datac