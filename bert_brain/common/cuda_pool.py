import time
import itertools
import gc
from functools import partial
from contextlib import contextmanager
import queue
from tqdm import tqdm
from threading import Thread
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch


__all__ = ['cuda_map_unordered',
           'cuda_pool_executor',
           'cuda_most_free_device',
           'cuda_memory_info',
           'DeviceMemoryInfo',
           'ProgressContext',
           'cuda_auto_empty_cache_context']


@contextmanager
def cuda_auto_empty_cache_context(device):

    if not isinstance(device, torch.device):
        device = torch.device(device)

    if device.type == 'cpu':
        yield device
    else:
        with torch.cuda.device(device) as device_context:
            yield device_context
            gc.collect()
            