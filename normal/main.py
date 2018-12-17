from bubble_sort import BubbleSort
import time

if __name__ == '__main__':
    b_sort = BubbleSort()
    arr = [54, 45, 4, 3, 2, 1, 34, 35, 56, 7, 6, 76, 7, 67, 6, 7, 56, 5, 56, 6, 5, 65, 656, 56]
    start = time.time()
    b_sort.bubble_sort(arr)
    end = time.time()
    print(arr, f'time: {end - start} seconds')
