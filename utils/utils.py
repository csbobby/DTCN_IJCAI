import pandas as pd
import numpy as np
from memory_profiler import profile

import sys
import datetime
import pdb
from utils.read_write_file import write_array_to_file
from utils.read_write_file import read_file_as_array
from utils.generate_h5 import check_h5, write_into_h5py, read_from_h5
import argparse
import os
print(os.environ['KERAS_BACKEND'])
import numpy as np

def cart_production(x,y):
    return [[x0, y0] for x0 in x for y0 in y]

def transform_info(info):
    if info.__class__==dict:
        flatten = np.arange(start = info['range'][0],stop = info['range'][1]+info['step'], step= info['step'])
    elif info.__class__==list:
        flatten = info
    else:
        raise Exception("Only Dict or list is allowed in extra paramter types!")
    return flatten

def flatten(item):
    if not isinstance(item, list):
        return [item]
    return sum(map(flatten, item), [])

def generate_param_matrix(extra_info):
    all_params = []
    for key in reversed(extra_info.keys()):
        if len(all_params)==0:
            all_params = transform_info(extra_info[key])
        else:
            all_params = cart_production(all_params,transform_info(extra_info[key]))
    final_params = []
    for row in all_params:
        row = flatten(row)
        final_row = {}
        for idx in range(0,len(extra_info.keys())):
            final_row[list(reversed(extra_info.keys()))[idx]] = row[idx]
        final_params.append(final_row)
    return final_params
