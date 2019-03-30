import numpy as np
import pandas as pd

class GenerateData():
    def __init__(self, size):
        self.size = size

    def gen_int(self):
        return list(np.random.choice(self.size, self.size, replace=False))

    def gen_float(self):
        return list(np.random.choice(float(self.size), self.size, replace=False))

    def write_dataset_csv(arrays, sorted_arrays, filename):
        dataset = {'unsorted': arrays, 'sorted': sorted_arrays}
        df = pd.DataFrame(dataset, columns=['unsorted', 'sorted'])
        df.to_csv(filename, sep='\t', encoding='utf-8')

    def read_dataset(filename):
        df = pd.read_csv(filename, sep='\t', encoding='utf-8')
        return df['unsorted'].values.T.tolist(), df['sorted'].values.T.tolist()
