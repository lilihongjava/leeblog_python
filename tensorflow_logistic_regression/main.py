# encoding: utf-8
"""
@author: lee
@time: 2020/6/24 16:35
@file: main.py
@desc: 
"""
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.utils.np_utils import to_categorical

from util.common_util import arg_check_transformation, multiple_gpu_strategy


def model_builder(x_data, class_num):
    if class_num == 2:  # 逻辑回归二分类
        layer0 = tf.keras.layers.Dense(1, input_shape=(x_data.shape[1],), activation='sigmoid')
        model = tf.keras.Sequential([layer0])
        model.compile(loss='binary_crossentropy', optimizer='adam')  # 这里用二元的交叉熵作为二分类的损失函数
    else:  # 多分类
        layer0 = tf.keras.layers.Dense(class_num, input_shape=(x_data.shape[1],), activation='softmax')
        model = tf.keras.Sequential([layer0])
        model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


def tf_logistic_regression(feature_column=None, label_column=None, class_num=None, gpu=None, input1=None,
                           output1=None):
    print("输入参数：", locals())
    feature_column = arg_check_transformation("list_name_str", "feature_column", feature_column)
    label_column = arg_check_transformation("list_name_str", "label_column", label_column)
    class_num = arg_check_transformation("int", "class_num", class_num)
    df = pd.read_csv(input1)
    try:
        x_data = df[feature_column]
        y_data = df[label_column]
        if class_num != 2 and y_data.shape[1] == 1:
            y_data = to_categorical(y_data)  # 一维的分类转成多列
            y_data = pd.DataFrame(y_data)
        if gpu:
            dataset, BATCH_SIZE, strategy = multiple_gpu_strategy(x_data, y_data)
            with strategy.scope():
                model = model_builder(x_data, class_num)
            model.fit(dataset.batch(BATCH_SIZE), verbose=False)
        else:
            model = model_builder(x_data, class_num)
            model.fit(x_data, y_data, epochs=1000)
    except Exception:
        raise Exception("模型训练错误")
    print("模型训练完成")
    if output1:
        model.save(output1)


if __name__ == '__main__':
    # 二分类
    tf_logistic_regression(feature_column="sepal length (cm),sepal width (cm),petal length (cm),petal width (cm)",
                           label_column="target", class_num=2,
                           gpu=False, input1="./data/iris_two.csv", output1="./data/output.h5")
    # 多分类
    tf_logistic_regression(feature_column="sepal length (cm),sepal width (cm),petal length (cm),petal width (cm)",
                           label_column="target", class_num=3,
                           gpu=False, input1="./data/iris.csv", output1="./data/output.h5")
