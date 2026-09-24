"""Execution-only limits for copairs; numerical routines and pairing rules unchanged."""

from contextlib import contextmanager
from multiprocessing.pool import ThreadPool

from copairs import compute
from threadpoolctl import threadpool_limits


@contextmanager
def bounded_copairs(workers):
    if workers < 1:
        raise ValueError("copairs workers must be positive")
    original = compute.ThreadPool
    # copairs parallel_map otherwise uses os.cpu_count(), ignoring CLI max_workers.
    compute.ThreadPool = lambda processes=None, *args, **kwargs: ThreadPool(
        min(workers, processes or workers), *args, **kwargs
    )
    try:
        with threadpool_limits(limits=1, user_api="blas"):
            yield
    finally:
        compute.ThreadPool = original
