import h5py
import numpy as np
from termcolor import colored
from tqdm import tqdm


def check_h5(file_path):
    print(colored('checking h5 file', 'green'))
    try:
        h5_file = h5py.File(file_path, "r")
        for key in h5_file.keys():
            print(colored("SHAPE of {} is {}".format(key, h5_file[key].shape), 'green'))
        h5_file.close()
        return True
    except:
        print(colored('h5 file doesn\'t exist', 'green'))
        return False

# file_path, key, value, key1, value1, key2, value2...
def write_into_h5py(file_path, *args):
    print(colored("Writing into H5, please wait", 'cyan'))
    mode = 'w'
    if check_h5(file_path):
        mode = 'r+'
    if len(args) % 2 != 0:
        raise ValueError("key must equal to values!")
    key = ""
    h5_file = h5py.File(file_path, mode)
    for index, element in tqdm(enumerate(args)):
        if index % 2 == 0:
            key = element
        else:
            h5_file.create_dataset(key, data=element)
    h5_file.flush()
    h5_file.close()





def get_key_shape(file_path, key):
    h5 = h5py.File(file_path, "r")
    shape = h5[key].shape
    h5.close()
    return shape


def get_h5py_file(file_path):
    return h5py.File(file_path, 'r')


def read_from_h5(file_path, key):
    h5 = h5py.File(file_path, "r")
    value = None
    if key in h5.keys():
        value = h5[key].value
    h5.close()
    return value


def read_partial_from_h5(file_path, key, start, end):
    h5 = h5py.File(file_path, "r")
    value = None
    if key in h5.keys():
        value = h5[key][start: end]
    h5.close()
    return value


# test
# key1 = 'key1'
# value1 = np.array([0, 1, 2, 3, 4, 5])
# key2 = 'key2'
# value2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# write_into_h5py('h5_data/test.hdf5', key1, value1, key2, value2)
# result = read_from_h5('h5_data/test.hdf5', key1)
# check_h5('h5_data/test.hdf5')
