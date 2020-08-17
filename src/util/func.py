import concurrent
from concurrent.futures.thread import ThreadPoolExecutor

from tqdm import tqdm

from util.iterable import chunk


def flatmap(func, iterable):
    def f(x):
        return list(func(x))

    return sum(list(map(f, iterable)), [])


def parallel_map(func, iterable, n_threads):
    if n_threads <= 1:
        return list(map(func, tqdm(iterable)))

    partitioned_list = chunk(list(iterable), n_threads)
    futures, results = [], []
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        for thread_idx, thread_list in enumerate(partitioned_list):
            if thread_idx == 0:
                thread_list = tqdm(thread_list)

            future = executor.submit(lambda xs: list(map(func, xs)), thread_list)
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            results.extend(future.result())

    return results


class Lazy:
    def __init__(self, factory_method):
        self.factory_method = factory_method
        self.data = None

    def __call__(self):
        return self.get()

    def get(self):
        if self.data is None:
            self.data = self.factory_method()

        return self.data
