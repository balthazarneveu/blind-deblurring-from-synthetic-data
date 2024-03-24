# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 20:59:03 2024

@author: jamyl
"""
from rstor.data.synthetic_dataloader import DeadLeavesDataset, DeadLeavesDatasetGPU
from time import perf_counter
import numba


def test_gpu_vs_cpu_dataloader():
    if not numba.cuda.is_available():
        return

    n = 10
    print("\n")

    print("=== Dead leaves with reversing")
    dataset = DeadLeavesDatasetGPU(number_of_circles=256, reverse=True)
    t1 = perf_counter()
    for i in range(n):
        _ = dataset[i]
    print(f"Mean time on {n} samples (numba) : {(perf_counter()-t1)/n}")

    dataset = DeadLeavesDataset(number_of_circles=256, reverse=True)
    t1 = perf_counter()
    for i in range(n):
        _ = dataset[i]
    print(f"Mean time on {n} samples (cv2): {(perf_counter()-t1)/n}")

    print("=== Dead leaves without reversing")
    dataset = DeadLeavesDatasetGPU(number_of_circles=256, reverse=False)
    t1 = perf_counter()
    for i in range(n):
        _ = dataset[i]
    print(f"Mean time on {n} samples (numba) : {(perf_counter()-t1)/n}")

    dataset = DeadLeavesDataset(number_of_circles=256, reverse=False)
    t1 = perf_counter()
    for i in range(n):
        _ = dataset[i]
    print(f"Mean time on {n} samples (cv2): {(perf_counter()-t1)/n}")
