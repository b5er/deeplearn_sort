import sys
sys.path.append('convolutional')
sys.path.append('data')
sys.path.append('normal_sort_algos')
sys.path.append('recurrent')
sys.path.append('vanilla')


CONVOLUTIONAL = True
DATA_GEN = False
N_SORT_ALGOS = False
VANILLA = False
RECURRENT = False


if __name__ == '__main__':

    if CONVOLUTIONAL:
        import convolutional.main

    if DATA_GEN:
        import data.main

    if N_SORT_ALGOS:
        import normal_sort_algos.main

    if VANILLA:
        import vanilla.main

    if RECURRENT:
        import recurrent.main
