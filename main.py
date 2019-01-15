import sys
sys.path.append('..')

import data
import normal_sort_algos
import vanilla.main
import recurrent
import convolutional

DATA_GEN = False
N_SORT_ALGOS = False
VANILLA = True
RECURRENT = False
CONVOLUTIONAL = False


if __name__ == '__main__':

    if DATA_GEN:
        data.main.main()

    if N_SORT_ALGOS:
        normal_sort_algos.main()

    if VANILLA:
        vanilla.main.main()

    if RECURRENT:
        recurrent.main()

    if CONVOLUTIONAL:
        convolutional.main()
