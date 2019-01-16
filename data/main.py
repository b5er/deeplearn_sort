from generate_data import GenerateData
from copy import deepcopy

GEN_TRAINING = True
GEN_TEST = True


def main(size, start, end, write_training, write_test):
    if GEN_TRAINING:
        # generate training dataset
        data = GenerateData(size, start, end)
        dataset = [data.gen_int() for _ in range(50000)]
        sorted_dataset = deepcopy(dataset)
        for data in sorted_dataset:
            data.sort()
        GenerateData.write_dataset_csv(dataset, sorted_dataset, write_training)

    if GEN_TEST:
        # generate testing dataset
        data = GenerateData(size, start, end)
        dataset = [data.gen_int() for _ in range(10000)]
        sorted_dataset = deepcopy(dataset)
        for data in sorted_dataset:
            data.sort()
        GenerateData.write_dataset_csv(dataset, sorted_dataset, write_test)


if __name__ == '__main__':
    size = 45
    start = 1
    end = 45
    write_training = f'../data/numeric/1_{size}/train_dataset.csv'
    write_test = f'../data/numeric/1_{size}/test_dataset.csv'
    main(size, start, end, write_training, write_test)
else:
    size = 45
    start = 1
    end = 45
    write_training = f'./data/numeric/1_{size}/train_dataset.csv'
    write_test = f'./data/numeric/1_{size}/test_dataset.csv'
    main(size, start, end, write_training, write_test)