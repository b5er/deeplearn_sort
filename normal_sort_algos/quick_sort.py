import random

class QuickSort():
    def __init__(self):
        pass

    def quick_sort(self, low, high, arr=[]):
        if len(arr) < 2 or not arr:
            return

        if low < high:
            pi = self.partition(arr, low, high)

            self.quick_sort(low, pi - 1, arr)
            self.quick_sort(pi + 1, high, arr)


    def partition(self, arr, low, high):
        i = low - 1
        pivot = arr[high]

        for j in range(low, high): # low = 0, high = 3
            if arr[j] <= pivot: # pivot = 4
                i += 1
                arr[i], arr[j] = arr[j], arr[i]

        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return i + 1
