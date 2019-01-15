class BubbleSort():
    def __init__(self):
        pass

    def bubble_sort(self, arr=[]):
        if len(arr) < 2 or not arr:
            return

        for i in range(len(arr) - 1, -1, -1):
            for j in range(i):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
