from generate_data import GenerateData

GEN_TRAINING = True
GEN_TEST = True

DATASET_SIZE = [45, 100, 1000]

def main(size, train_dataset_path, test_dataset_path):
    if GEN_TRAINING:
        data = GenerateData(size)
        dataset = [data.gen_int() for _ in range(50000)]
        sorted_dataset = list(map(sorted, dataset))
        GenerateData.write_dataset_csv(dataset, sorted_dataset, train_dataset_path)

    if GEN_TEST:
        data = GenerateData(size)
        dataset = [data.gen_int() for _ in range(10000)]
        sorted_dataset = list(map(sorted, dataset))
        GenerateData.write_dataset_csv(dataset, sorted_dataset, test_dataset_path)


if __name__ == '__main__':
    for size in DATASET_SIZE:
        train_dataset_path = f'../data/numeric/{size}/train_dataset.csv'
        test_dataset_path = f'../data/numeric/{size}/test_dataset.csv'
        main(size, train_dataset_path, test_dataset_path)
else:
    for size in DATASET_SIZE:
        train_dataset_path = f'../data/numeric/{size}/train_dataset.csv'
        test_dataset_path = f'../data/numeric/{size}/test_dataset.csv'
        main(size, train_dataset_path, test_dataset_path)
