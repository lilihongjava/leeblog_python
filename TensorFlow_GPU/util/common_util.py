# encoding: utf-8
"""
@author: lee
@file: common_util.py
@desc: 
"""
import tensorflow as tf


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


def arg_check_transformation(arg_type, arg_name, val):
    if val == 'null':
        val = None
        return val
    return switcher.get(arg_type)(arg_type, arg_name, val)
