import random
import pandas as pd

class GenerateData():
    def __init__(self, size, start=0, end=100):
        self.size = size
        self.start = start
        self.end = end

    def gen_int(self):
        return [(x + random.randint(self.start, self.end)) * random.randint(self.start, self.end) for x in range(self.size)]

    def gen_float(self):
        return [(x + random.random()) * random.random() for x in range(self.size)]

    def quick_sort(self, array):
        pass
