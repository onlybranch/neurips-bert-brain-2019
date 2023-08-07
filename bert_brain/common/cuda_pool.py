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
                for _ in range(max_workers):  # signal workers waiting on a device to exit
                    device_queue.put(-1)
                return
            else:
                needed_count += 1
        except queue.Empty:
            pass

        if needed_count > 0:
            memory_info = cuda_memory_info()
            selected_free = None
            selected_device = None
            for device_id, device_memory in enumerate(memory_info):
                projected_use = current_use[device_id] if device_id in current_use else 0
                device_free = device_memory.free + torch.cuda.memory_allocated(device_id) - projected_use
                if device_free > min_memory and (selected_free is None or device_free > selected_free):
                    selected_free = device_free
                    selected_device = device_id

            if selected_device is not None:
                if selected_device not in current_use:
                    current_use[selected_device] = min_memory
                else:
                    current_use[selected_device] += min_memory
                device_queue.put(selected_device)
                needed_count -= 1


def _cuda_memory_retry_wrap(retry_item):
    try:
        args = () if retry_item.args is None else retry_item.args
        retry_item.result = retry_item.func(*args)
        retry_item.exception = None
    except RuntimeError as e:
        if str(e) != 'CUDA error: out of memory':
            raise
        retry_item.result = None
        retry_item.exception = e

    return retry_item


class OutOfMemoryRetry(object):

    def __init__(self, func, args):
        self.func = func
        self.args = args
        self.num_tries = 0
        self.result = None
        self.exception = None


_progress_stop_sentinel = 'kill_progress_monitor'


def _monitor_progress(progress_t, progress_queue):
    while True:
        p = progress_queue.get()
        if p == _progress_stop_sentinel:
            return
        if p < 0:
            progress_t.n = progress_t.n - p
            progress_t.refresh()
        else:
            progress_t.update(p)


class ProgressContext(object):

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self._mp_context = None
        self._progress_queue = None
        self._progress_bar = None

    @property
    def mp_context(self):
        return self._mp_context

    @property
    def progress_queue(self):
        return self._progress_queue

    @property
    def progress_bar(self):
        return self._progress_bar

    def __enter__(self):
        self._mp_context = get_context('spawn')
        self._progress_queue = self.mp_context.Queue()
        self._progress_bar = tqdm(*self._args, **self._kwargs)
        self._args = None
        self._kwargs = None
        progress_monitor = Thread(target=_monitor_progress, args=(self.progress_bar, self.progress_queue), daemon=True)
        progress_monitor.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.progress_bar.close()
        self.progress_queue.put(_progress_stop_sentinel)


def cuda_map_unordered(
        min_memory, func, iterables,
        max_workers=None, mp_context=None, initializer=None, initargs=None,
        num_cuda_memory_retries=0, chunksize=1):

    items = iterables

    if num_cuda_memory_retries > 0:
        items = map(lambda args: OutOfMemoryRetry(func, args), zip(*items))

    finished = False

    while not finished:
        retries = list()

        with cuda_pool_executor(
                min_memory, max_workers, mp_context=mp_context, initializer=initializer, initargs=initargs) as ex:

            if num_cuda_memory_retries > 0:
                result = ex.map(_cuda_memory_retry_wrap, items, chunksize=chunksize)
                for item in result:
                    if item.exception is not None:
                        if item.num_tries < num_cuda_memory_r