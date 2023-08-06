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
            torch.cuda.empty_cache()


def _set_device_id_and_initialize(device_queue, no_available_queue, initializer, initargs):

    try:
        device_id = device_queue.get(timeout=5)
    except queue.Empty:
        no_available_queue.put(1)
        device_id = device_queue.get()

    if device_id < 0:
        return

    print('binding to device {}'.format(device_id))

    torch.cuda.set_device(device_id)

    if initializer is not None:
        if initargs is None:
            initargs = ()
        initializer(*initargs)


def _monitor_devices(max_workers, min_memory, starting_ids, device_queue, no_available_queue):

    current_use = dict()
    for device_id in starting_ids:
        if device_id not in current_use:
            current_use[device_id] = min_memory
        else:
            current_use[device_id] += min_memory

    needed_count = 0
    while True:
        try:
            need = no_available_queue.get(timeout=100)
            if need == 2:  # signals shutdown
                for _ in range(max_workers):  # si