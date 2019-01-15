import sys
sys.path.append('..')

from vanilla_mlp import MLP
from data.generate_data import GenerateData
import tensorflow as tf
import numpy as np
import pandas as pd
import ast
import math
import time


train_model = True
eval_model = False
predict = False


def write_result():
    pass


def normalize(x, min, max):
    if(max - min == 0):
        return -1
    return (x - min) / (max - min)

def denormalize(y, min, max):
    return (y * (max - min)) + min



if __name__ == '__main__':

    size = 45

    if train_model:
        input, output = GenerateData.read_dataset(f'../data/numeric/1_{size}/train_dataset.csv')
        input = [ast.literal_eval(elem) for elem in input]
        list_max = len(input[0])
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

        neural_net = MLP(25, 5, batch=10, cont=False, epochs=1000, size=size)
        neural_net.build(len(input[0]))
        neural_net.train(normal_train_input, normal_train_output)

    if eval_model:
        model = tf.keras.models.load_model(f'./models/sort_net_{size}.mpl')

        input, output = GenerateData.read_dataset(f'../data/numeric/1_{size}/test_dataset.csv')
        list_max = len(input[0])
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
        model = tf.keras.models.load_model('./models/sort_net.mpl')
        data = [[36, 36, 36, 14, 39, 28, 30, 44, 5, 7, 32, 39, 16, 21, 22, 37, 23, 2, 34, 17, 28, 7, 33, 22, 4, 16, 36, 6, 18, 11, 35, 4, 1, 5, 42, 41, 37, 36, 26, 12, 5, 28, 10, 36, 25]]
        # [1, 2, 4, 4, 5, 5, 5, 6, 7, 7, 10, 11, 12, 14, 16, 16, 17, 18, 21, 22, 22, 23, 25, 26, 28, 28, 28, 30, 32, 33, 34, 35, 36, 36, 36, 36, 36, 36, 37, 37, 39, 39, 41, 42, 44] Sorted
        list_max = len(data[0])
        list_min = min(data[0])
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
