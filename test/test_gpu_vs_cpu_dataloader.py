# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 20:59:03 2024

@author: jamyl
"""
from rstor.data.dataloader import DeadLeavesDataset, DeadLeavesDatasetGPU
from time import perf_counter
import numba


def test_gpu_vs_cpu_dataloader():
    if not numba.cuda.is_available():
        return

    n = 20
    print("\n")

    dataset = DeadLeavesDatasetGPU()
    t1 = perf_counter()
    for i in range(n):
        _ = dataset[i]
    print(f"Mean time on {n} samples (numba) : {(perf_counter()-t1)/n}")

    dataset = DeadLeavesDataset()
    t1 = perf_counter()
    for i in range(n):
        _ = dataset[i]
    print(f"Mean time on {n} samples (cv2): {(perf_counter()-t1)/n}")
