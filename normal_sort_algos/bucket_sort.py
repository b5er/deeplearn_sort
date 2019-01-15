import random
import math

class BucketSort():
    def __init__(self):
        pass

    def bucket_sort(self, arr=[]):
        if len(arr) < 2 or not arr:
            return

        code = self.hashing(arr)
        buckets = [list() for _ in range(code[1])]

		# distribute elements into each bucket
        for num in arr:
            x = self.re_hashing(num, code)
            buck = buckets[x]
            buck.append(num)

		# sort each bucket
        for bucket in buckets:
            self.insertion_sort(bucket)

		# merge buckets
        index = 0
        for bucket in range(len(buckets)):
            for value in buckets[bucket]:
                arr[index] = value
                index += 1

    def hashing(self, arr):
        m = arr[0]
        for i in range(1, len(arr)):
            if m < arr[i]:
                m = arr[i]
            result = [m, int(math.sqrt(len(arr)))]
        return result

    def re_hashing(self, num, code):
        return int(num / code[0] * (code[1] - 1))

    def insertion_sort(self, arr):
        if len(arr) < 2:
            return

        for i in range(1, len(arr)):
            curr = arr[i]
            j = i - 1
            while j >= 0 and arr[j] > curr:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = curr
