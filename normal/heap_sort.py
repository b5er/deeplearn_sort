class HeapSort():
    def __init__(self):
        pass

    def heap_sort(self, arr=[]):
        n = len(arr)

        if n < 2 or not arr:
            return

        for i in range(n, -1, -1):
            self.heapify(arr, n, i)

        for i in range(n - 1, 0, -1):
            arr[i], arr[0] = arr[0], arr[i]
            self.heapify(arr, i, 0)

    def heapify(self, arr, n, i):
        largest = i
        left = (2 * i) + 1
        right = (2 * i) + 2

        if left < n and arr[i] < arr[left]:
            largest = left

        if right < n and arr[largest] < arr[right]:
            largest = right

        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            self.heapify(arr, n, largest)
