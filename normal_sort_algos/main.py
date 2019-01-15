import sys
sys.path.append('..')

from data.generate_data import GenerateData
from copy import deepcopy
import ast

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



if __name__ == '__main__':

    size = 45
    input, output = GenerateData.read_dataset(f'../data/numeric/1_{size}/train_dataset.csv')
    input, output = ast.literal_eval(input[0]), ast.literal_eval(output[0])

    if BUBBLE_SORT:
        arr = deepcopy(input)
        bubb_sort = BubbleSort()
        start = time.time()
        bubb_sort.bubble_sort(arr)
        end = time.time()
        assert arr == output
        print('Bubble Sort:', f'time: {end - start} seconds')

    if BUCKET_SORT:
        arr = deepcopy(input)
        buck_sort = BucketSort()
        start = time.time()
        buck_sort.bucket_sort(arr)
        end = time.time()
        assert arr == output
        print('Bucket Sort:', f'time: {end - start} seconds')

    if HEAP_SORT:
        arr = deepcopy(input)
        h_sort = HeapSort()
        start = time.time()
        h_sort.heap_sort(arr)
        end = time.time()
        assert arr == output
        print('Heap Sort:', f'time: {end - start} seconds')

    if MERGE_SORT:
        arr = deepcopy(input)
        m_sort = MergeSort()
        start = time.time()
        m_sort.merge_sort(arr)
        end = time.time()
        assert arr == output
        print('Merge Sort:', f'time: {end - start} seconds')

    if QUICK_SORT:
        arr = deepcopy(input)
        q_sort = QuickSort()
        start = time.time()
        q_sort.quick_sort(0, len(arr) - 1, arr)
        end = time.time()
        assert arr == output
        print('Quick Sort:', f'time: {end - start} seconds')
