import random
import pandas as pd

class GenerateData():
    def __init__(self, size, start=0, end=100):
        self.size = size
        self.start = start
        self.end = end

    def gen_int(self):
        return [random.randint(self.start, self.end) for x in range(self.size)]

    def gen_float(self):
        return [(x + random.random()) * random.random() for x in range(self.size)]

    def write_dataset_csv(arrays, sorted_arrays, filename):
        dataset = {'unsorted': arrays, 'sorted': sorted_arrays}
        df = pd.DataFrame(dataset, columns=['unsorted', 'sorted'])
        df.to_csv(filename, sep='\t', encoding='utf-8')

    def read_dataset(filename):
        df = pd.read_csv(filename, sep='\t', encoding='utf-8')
        return df['unsorted'].values.T.tolist(), df['sorted'].values.T.tolist()
