# encoding: utf-8
"""
@author: lee
@file: common_util.py
@desc: 
"""


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
