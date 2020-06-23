# encoding: utf-8
"""
@author: lee
@time: 2020/6/22 16:04
@file: main.py
@desc: 
"""
import pandas as pd
import tensorflow as tf

from util.common_util import arg_check_transformation


def model_train(x_data, y_data):
    layer0 = tf.keras.layers.Dense(1, input_shape=(x_data.shape[1],))
    model = tf.keras.Sequential([layer0])
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_data, y_data, epochs=100, verbose=False)
    return model


def tf_linear_regression(feature_column=None, label_column=None, gpu=None, input1=None,
                         output1=None):
    """
    :param feature_column:  特征列
    :param label_column:  标签列
    :param gpu:  是否开启gpu
    :param input1:  输入数据位置
    :param output1:  模型输出位置
    :return: 
    """
    print("输入参数：", locals())
    feature_column = arg_check_transformation("list_name_str", "feature_column", feature_column)
    label_column = arg_check_transformation("list_name_str", "label_column", label_column)
    df = pd.read_csv(input1)
    try:
        x_data = df[feature_column]
        y_data = df[label_column]
        if gpu:
            tf.debugging.set_log_device_placement(True)
            # 多卡gpu支持，维度必须是gpu卡的倍数
            gpu_len = len(tf.config.experimental.list_physical_devices('GPU'))
            print("gpu_len:" + str(gpu_len))
            dataset = tf.data.Dataset.from_tensor_slices((x_data.values, y_data.values))
            strategy = tf.distribute.MirroredStrategy()
            BATCH_SIZE_PER_REPLICA = 64
            BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
            print("x_data shape:" + str(x_data.shape))
            # tf1.14.0版本 维度必须是gpu卡的倍数 if x_data.shape[1] % gpu_len == 0 and x_data.shape[0] % gpu_len == 0:
            print("执行多卡gpu")
            with strategy.scope():
                layer0 = tf.keras.layers.Dense(1, input_shape=(x_data.shape[1],))
                model = tf.keras.Sequential([layer0])
                model.compile(loss='mean_squared_error', optimizer='adam')
            model.fit(dataset.batch(BATCH_SIZE), verbose=False)
        else:
            model = model_train(x_data, y_data)
    except Exception:
        raise Exception("模型训练错误")
    print("模型训练完成")
    if output1:
        model.save(output1)


if __name__ == '__main__':
    tf_linear_regression(feature_column="CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B", label_column="MEDV",
                         gpu=True, input1="./data/boston.csv", output1="./data/output.h5")
