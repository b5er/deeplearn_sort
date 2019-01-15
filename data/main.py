from generate_data import GenerateData
from copy import deepcopy

GEN_TRAINING = True
GEN_TEST = True

if __name__ == '__main__':

    size = 45
    start = 1
    end = 45

    if generate_training:
        # generate training dataset
        data = GenerateData(size, start, end)
        dataset = [data.gen_int() for _ in range(50000)]
        sorted_dataset = deepcopy(dataset)
        for data in sorted_dataset:
            data.sort()
        write_dataset_csv(dataset, sorted_dataset, '../data/numeric/1_45/train_dataset.csv')

    if generate_test:
        # generate testing dataset
        data = GenerateData(size, start, end)
        dataset = [data.gen_int() for _ in range(10000)]
        sorted_dataset = deepcopy(dataset)
        for data in sorted_dataset:
            data.sort()
        write_dataset_csv(dataset, sorted_dataset, '../data/numeric/1_45/test_dataset.csv')
