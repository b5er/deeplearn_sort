import numpy as np
import pandas as pd
import os

class GenerateData():
    def __init__(self, size):
        self.size = size

    def gen_int(self):
        return list(np.random.choice(self.size, self.size, replace=False))

    def gen_float(self):
        return list(np.random.choice(float(self.size), self.size, replace=False))

    def write_dataset_csv(arrays, sorted_arrays, file_path):
        dataset = {'unsorted': arrays, 'sorted': sorted_arrays}
        df = pd.DataFrame(dataset, columns=['unsorted', 'sorted'])
        try:
            df.to_csv(file_path, sep='\t', encoding='utf-8')
        except:
            if not os.path.exists(file_path):
                print(f'Unabled to find {file_path}\nCreating...')
                end_dir = len(file_path) - 1
                for i in (range(len(file_path))):
                    if file_path[~i] == '/':
                        break
                    end_dir -= 1
                dir_path = file_path[:end_dir]
                os.makedirs(dir_path)
            df.to_csv(file_path, sep='\t', encoding='utf-8')

    def read_dataset(filename):
        df = pd.read_csv(filename, sep='\t', encoding='utf-8')
        return df['unsorted'].values.T.tolist(), df['sorted'].values.T.tolist()
