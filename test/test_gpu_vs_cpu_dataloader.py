# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 20:59:03 2024

@author: jamyl
"""
from rstor.data.dataloader import DeadLeavesDataset, DeadLeavesDatasetGPU
from time import perf_counter


def test_gpu_vs_cpu_dataloader():
    n = 20
    print("\n")

    dataset = DeadLeavesDatasetGPU()
    t1 = perf_counter()
    for _ in range(n):
        a = dataset[_]
    print(f"Mean time on {n} samples (numba) : {(perf_counter()-t1)/n}")

    dataset = DeadLeavesDataset()
    t1 = perf_counter()
    for _ in range(n):
        a = dataset[_]
    print(f"Mean time on {n} samples (cv2): {(perf_counter()-t1)/n}")
 