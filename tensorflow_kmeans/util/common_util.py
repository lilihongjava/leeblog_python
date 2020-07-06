# encoding: utf-8
"""
@author: lee
@file: common_util.py
@desc: 
"""
import json
import math
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from io import StringIO


def int_arg_check_transformation(arg_type, arg_name, val):
    try:
        val = int(val)
    except ValueError:
        raise ValueError("参数%s的值%s需要是%s类型" % (arg_name, val, arg_type))
    return val


def float_arg_check_transformation(arg_type, arg_name, val):
    try:
        val = float(val)
    except ValueError:
        raise ValueError("参数%s的值%s需要是%s类型" % (arg_name, val, arg_type))
    return val


def list_int_arg_check_transformation(arg_type, arg_name, val):
    try:
        val = list(map(int, val.split(",")))
    except ValueError:
        raise ValueError("参数%s的值%s需要是类似1,2,3这样的结构" % (arg_name, val))
    return val


def boolean_int_arg_check_transformation(arg_type, arg_name, val):
    try:
        val = int(val)
        if val == 1:
            val = True
        else:
            val = False
    except ValueError:
        raise ValueError("参数%s的值%s需要是0或者1，1表达true，0代表false" % (arg_name, val))
    return val


def list_name_str_arg_check_transformation(arg_type, arg_name, val):
    try:
        val = list(val.split(','))
    except ValueError:
        raise ValueError("参数%s的值%s需要是类似name1,name2,name3这样的结构" % (arg_name, val))
    return val


switcher = {
    "int": int_arg_check_transformation,
    "float": float_arg_check_transformation,
    "list_int": list_int_arg_check_transformation,
    "boolean_int": boolean_int_arg_check_transformation,
    "list_name_str": list_name_str_arg_check_transformation
}


def arg_check_transformation(arg_type, arg_name, val):
    if val == 'null':
        val = None
        return val
    return switcher.get(arg_type)(arg_type, arg_name, val)


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


class Report:
    def __init__(self, content):
        self.content = content


def create_df(forecast_data):
    df = pd.DataFrame()
    for index, i_data in enumerate(forecast_data):
        if index == 0:
            continue
        temp = pd.read_csv(StringIO(i_data), thousands=',', sep=',', header=None,
                           names=list(forecast_data[0].split(',')))
        df = df.append(temp, ignore_index=True)
    return df


def df_to_str(data_frame):
    df_str = data_frame.to_string(index=False).split('\n')
    vals = [','.join(ele.split()) for ele in df_str]
    return vals


def multiple_gpu_strategy(x_data, y_data):
    tf.debugging.set_log_device_placement(True)
    # tf.config.set_soft_device_placement(True)  # 自动选择一个gpu
    gpu_len = len(tf.config.experimental.list_physical_devices('GPU'))
    print("gpu_len:" + str(gpu_len))
    dataset = tf.data.Dataset.from_tensor_slices((x_data.values, y_data.values))
    strategy = tf.distribute.MirroredStrategy()
    BATCH_SIZE_PER_REPLICA = 64
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    print("x_data shape:" + str(x_data.shape))
    print("执行多卡gpu")
    return dataset, BATCH_SIZE, strategy


class ParasData:
    def __init__(self, feature, json):
        self.feature = feature
        self.json = json


class FeatureJson:
    def __init__(self, name, paras):
        self.name = name
        self.paras = paras
