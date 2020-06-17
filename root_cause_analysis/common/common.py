# encoding: utf-8
"""
@author: lee
@time: 2020/6/8 15:09
@file: common.py
@desc: 
"""


def cc_label(i):
    if i == '传输系统-传输设备':
        return 0
    elif i == '传输系统-光缆故障':
        return 1
    elif i == '传输系统-其他原因':
        return 2
    elif i == '动力环境-UPS':
        return 3
    elif i == '动力环境-电力部门供电':
        return 4
    elif i == '动力环境-电源线路故障':
        return 5
    elif i == '动力环境-动环监控系统':
        return 6
    elif i == '动力环境-动力环境故障':
        return 7
    elif i == '动力环境-高低压设备':
        return 8
    elif i == '动力环境-环境':
        return 9
    elif i == '动力环境-开关电源':
        return 10
    elif i == '其他-误告警或自动恢复':
        return 11
    elif i == '人为操作-告警测试':
        return 12
    elif i == '人为操作-工程施工':
        return 13
    elif i == '人为操作-物业原因':
        return 14
    elif i == '主设备-参数配置异常':
        return 15
    elif i == '主设备-其他':
        return 16
    elif i == '主设备-软件故障':
        return 17
    elif i == '主设备-设备复位问题':
        return 18
    elif i == '主设备-设备连线故障':
        return 19
    elif i == '主设备-天馈线故障':
        return 20
    elif i == '主设备-信源问题':
        return 21
    elif i == '主设备-硬件故障':
        return 22
