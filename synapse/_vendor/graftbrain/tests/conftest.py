import torch


def pytest_sessionstart(session):
    # Small CPU fixtures perform better without large OpenMP thread pools.
    torch.set_num_threads(2)
