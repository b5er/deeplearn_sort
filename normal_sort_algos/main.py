import sys
sys.path.append('..')

from data.generate_data import GenerateData
from threading import Thread
from copy import deepcopy
import ast
import os

from bubble_sort import BubbleSort
from bucket_sort import BucketSort
from heap_sort import HeapSort
from merge_sort import MergeSort
from quick_sort import QuickSort
import time

BUBBLE_SORT = True
BUCKET_SORT = True
HEAP_SORT = True
MERGE_SORT = True
QUICK_SORT = True

def fn_bubble_sort(input, output, start):
    bubb_sort = BubbleSort()
    bubb_sort.bubble_sort(input)
    assert input == output
    end = time.time()
    print(f'bubble sort: {end - start}s')

def fn_bucket_sort(input, output, start):
    buck_sort = BucketSort()
    buck_sort.bucket_sort(input)
    assert input == output
    end = time.time()
    print(f'bucket sort: {end - start}s')

def fn_heap_sort(input, output, start):
    heap_sort = HeapSort()
    heap_sort.heap_sort(input)
    assert input == output
    end = time.time()
    print(f'heap sort: {end - start}s')

def fn_merge_sort(input, output, start):
    m_sort = MergeSort()
    m_sort.merge_sort(input)
    assert input == output
    end = time.time()
    print(f'merge sort: {end - start}s')

def fn_quick_sort(input, output, start):
    q_sort = QuickSort()
    q_sort.quick_sort(0, len(input) - 1, input)
    assert input == output
    end = time.time()
    print(f'quick sort: {end - start}s')

def main(size, input, output):
    fn_sorts = []

    if BUBBLE_SORT:
        fn_sorts.append(fn_bubble_sort)
    if BUCKET_SORT:
        fn_sorts.append(fn_bucket_sort)
    if HEAP_SORT:
        fn_sorts.append(fn_heap_sort)
    if MERGE_SORT:
        fn_sorts.append(fn_merge_sort)
    if QUICK_SORT:
        fn_sorts.append(fn_quick_sort)

    start = time.time()
    for (i_data, o_data) in zip(input, output):
        i_data, o_data = ast.literal_eval(i_data), ast.literal_eval(o_data)
        for i in range(len(fn_sorts)):
            fn_sorts[i](deepcopy(i_data), deepcopy(o_data), time.time())
    end = time.time()

    print(f'total time: {end - start}s')


if __name__ == '__main__':
    sizes = [45, 100, 1000]
    for size in sizes:
        read_training = f'../data/numeric/{size}/train_dataset.csv'
        input, output = GenerateData.read_dataset(read_training)
        thread = Thread(target=main, args=(size, input, output))
        print('--------', thread.getName(), '----------')
        thread.start()
        thread.join()
else:
    sizes = [45, 100, 1000]
    for size in sizes:
        read_training = f'./data/numeric/{size}/train_dataset.csv'
        input, output = GenerateData.read_dataset(read_training)
        input, output = ast.literal_eval(input[0]), ast.literal_eval(output[0])
        main(size, read_training, input, output)
