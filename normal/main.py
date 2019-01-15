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

    arr = [12, 9, 20, 7, 5]

    if BUBBLE_SORT:
        bubb_sort = BubbleSort()
        start = time.time()
        bubb_sort.bubble_sort(arr)
        end = time.time()
        print('Bubble Sort::', arr, f'time: {end - start} seconds')

    arr = [12, 9, 20, 7, 5]

    if BUCKET_SORT:
        buck_sort = BucketSort()
        start = time.time()
        buck_sort.bucket_sort(arr)
        end = time.time()
        print('Bucket Sort::', arr, f'time: {end - start} seconds')

    arr = [12, 9, 20, 7, 5]

    if HEAP_SORT:
        h_sort = HeapSort()
        start = time.time()
        h_sort.heap_sort(arr)
        end = time.time()
        print('Heap Sort::', arr, f'time: {end - start} seconds')

    arr = [12, 9, 20, 7, 5]

    if MERGE_SORT:
        m_sort = MergeSort()
        start = time.time()
        m_sort.merge_sort(arr)
        end = time.time()
        print('Merge Sort::', arr, f'time: {end - start} seconds')

    arr = [12, 9, 20, 7, 5]

    if QUICK_SORT:
        q_sort = QuickSort()
        start = time.time()
        q_sort.quick_sort(0, len(arr) - 1, arr)
        end = time.time()
        print('Quick Sort::', arr, f'time: {end - start} seconds')
