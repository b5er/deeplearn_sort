import random

class MergeSort():
    def __init__(self):
        pass

    def merge_sort(self, arr=[]):
        if len(arr) < 2 or not arr:
            return

        if len(arr) < 2:
            return
        left = []
        right = []
        n = len(arr)

        mid = n // 2
        left = [arr[i] for i in range(mid)]
        right = [arr[i] for i in range(mid, n)]

        self.merge_sort(left)
        self.merge_sort(right)
        self.merge(left, right, arr)


    def merge(self, left, right, arr):
        i = j = k = 0

        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                arr[k] = left[i]
                i += 1
            else:
                arr[k] = right[j]
                j += 1
            k += 1

        while i < len(left):
            arr[k] = left[i]
            i += 1
            k += 1

        while j < len(right):
            arr[k] = right[j]
            j += 1
            k += 1
