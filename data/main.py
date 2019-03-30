from generate_data import GenerateData

GEN_TRAINING = True
GEN_TEST = True


def main(size, write_training, write_test):
    if GEN_TRAINING:
        data = GenerateData(size)
        dataset = [data.gen_int() for _ in range(50000)]
        sorted_dataset = list(map(sorted, dataset))
        GenerateData.write_dataset_csv(dataset, sorted_dataset, write_training)

    if GEN_TEST:
        data = GenerateData(size)
        dataset = [data.gen_int() for _ in range(10000)]
        sorted_dataset = list(map(sorted, dataset))
        GenerateData.write_dataset_csv(dataset, sorted_dataset, write_test)


if __name__ == '__main__':
    sizes = [45, 100, 1000]
    for size in sizes:
        write_training = f'../data/numeric/1_{size}/train_dataset.csv'
        write_test = f'../data/numeric/1_{size}/test_dataset.csv'
        main(size, write_training, write_test)
else:
    size = [45, 100, 1000]
    for size in sizes:
        write_training = f'./data/numeric/1_{size}/train_dataset.csv'
        write_test = f'./data/numeric/1_{size}/test_dataset.csv'
        main(size, write_training, write_test)
