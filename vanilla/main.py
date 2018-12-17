import sys
sys.path.append('..')

from generate_data import GenerateData
from mlp import MLP

import tensorflow as tf
import numpy as np
import pandas as pd
from copy import deepcopy
import ast
import math
import time

generate_data = False
train_model = False
eval_model = False
predict = True

def write_dataset_csv(arrays, sorted_arrays, filename):
    dataset = {'unsorted': arrays, 'sorted': sorted_arrays}
    df = pd.DataFrame(dataset, columns=['unsorted', 'sorted'])
    df.to_csv(filename, sep='\t', encoding='utf-8')

def read_dataset(filename):
    df = pd.read_csv(filename, sep='\t', encoding='utf-8')
    return df['unsorted'].values.T.tolist(), df['sorted'].values.T.tolist()

def normalize(x, min, max):
    if(max - min == 0):
        return -1
    return (x - min) / (max - min)


def denormalize(y, min, max):
    return (y * (max - min)) + min

if __name__ == '__main__':
    if generate_data:
        # generate training dataset
        data = GenerateData(5, 1, 5)
        dataset = [data.gen_int() for _ in range(50000)]
        sorted_dataset = deepcopy(dataset)
        for data in sorted_dataset:
            data.sort()
        write_dataset_csv(dataset, sorted_dataset, '../data/numeric/1_45/train_dataset.csv')

        # generate testing dataset
        data = GenerateData(5, 1, 5)
        dataset = [data.gen_int() for _ in range(10000)]
        sorted_dataset = deepcopy(dataset)
        for data in sorted_dataset:
            data.sort()
        write_dataset_csv(dataset, sorted_dataset, '../data/numeric/1_45/test_dataset.csv')

    if train_model:
        input, output = read_dataset('../data/numeric/1_45/train_dataset.csv')
        input = [ast.literal_eval(elem) for elem in input]
        list_max = 45
        list_min = 1
        normal_train_input = []
        for unsort in input:
            normalized_unsort = []
            for elem in unsort:
                normalized_unsort.append(normalize(elem, list_min, list_max))
            normal_train_input.append(normalized_unsort)

        output = [ast.literal_eval(elem) for elem in output]
        normal_train_output = []
        for sorted in output:
            normalized_sorted = []
            for elem in sorted:
                normalized_sorted.append(normalize(elem, list_min, list_max))
            normal_train_output.append(normalized_sorted)

        neural_net = MLP(25, 10, epochs=200000) # 16 --> 8 hours
        neural_net.build()
        neural_net.train(normal_train_input, normal_train_output)

    if eval_model:
        model = tf.keras.models.load_model("sort_net.mpl")

        input, output = read_dataset('../data/numeric/1_45/test_dataset.csv')
        list_max = 45
        list_min = 1
        input = [ast.literal_eval(elem) for elem in input]
        normal_test_input = []
        for unsort in input:
            normalized_unsort = []
            for elem in unsort:
                normalized_unsort.append(normalize(elem, list_min, list_max))
            normal_test_input.append(normalized_unsort)

        output = [ast.literal_eval(elem) for elem in output]
        normal_test_output = []
        for sorted in output:
            normalized_sorted = []
            for elem in sorted:
                normalized_sorted.append(normalize(elem, list_min, list_max))
            normal_test_output.append(normalized_sorted)

        total_scores = []
        loop_step = 0
        for test_input, test_output in zip(normal_test_input, normal_test_output):
            tf_test_input, tf_test_output = tf.constant([test_input], shape=[1,5]), tf.constant([test_output], shape=[1,5])
            score = model.evaluate(tf_test_input, tf_test_output, steps=1, verbose=0)
            print(loop_step, 'acc:', score[1]*100)
            total_scores.append(score[1]*100)
            if loop_step == 1500:
                break
            loop_step += 1
        print('\n\n')
        print('Mean:', np.mean(total_scores), 'Standard Dev:', np.std(total_scores))

    if predict:
        model = tf.keras.models.load_model("sort_net.mpl")
        data = [[4,3,6,10,8]] # [3, 4, 6, 8, 10] Sorted
        list_max = 45
        list_min = 1
        normal_pred = []
        for unsort in data:
            normalized_unsort = []
            for elem in unsort:
                normalized_unsort.append(normalize(elem, list_min, list_max))
            normal_pred.append(normalized_unsort)
        # print(normal_pred)
        unseen_data = tf.constant(normal_pred, shape=[1, 5])

        start = time.time()
        result = model.predict(unseen_data, steps=1)
        end = time.time()
        result = result.flatten()
        start2 = time.time()
        result2 = sorted(data[0])
        end2 = time.time()

        print([round(denormalize(elem, list_min, list_max)) for elem in result])
        print(f'time: {end - start} seconds')
        print(f'time: {end2 - start2} seconds')
