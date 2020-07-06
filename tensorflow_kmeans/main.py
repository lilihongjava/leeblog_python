# encoding: utf-8
"""
@author: lee
@time: 2020/6/29 10:41
@file: main.py
@desc: 
"""
import pandas as pd
import tensorflow as tf

import numpy as np

from tensorflow_kmeans.util.common_util import create_df
from tensorflow_kmeans.util.fileUtil import get_last_dir
from util.common_util import arg_check_transformation


def tf_k_means_model(feature_column=None, center_count=None, input1=None, output1=None):
    print("输入参数：", locals())
    feature_column = arg_check_transformation("list_name_str", "feature_column", feature_column)
    if center_count:
        center_count = arg_check_transformation("int", "center_count", center_count)
    else:
        raise Exception("聚类数不能为空")

    df = pd.read_csv(input1)

    model = tf.compat.v1.estimator.experimental.KMeans(
        num_clusters=center_count, use_mini_batch=False)

    points = np.array(df[feature_column])

    def input_fn():
        return tf.data.Dataset.from_tensors(tf.convert_to_tensor(points, dtype=tf.float32)).repeat(2)

    # train
    num_iterations = 10
    previous_centers = None
    for _ in range(num_iterations):
        model.train(input_fn)
        cluster_centers = model.cluster_centers()
        if previous_centers is not None:
            print('delta:', cluster_centers - previous_centers)
        previous_centers = cluster_centers
        print('score:', model.score(input_fn))
    print('cluster centers:', cluster_centers)

    if output1:
        my_feature_columns = []
        for key in feature_column:
            my_feature_columns.append(tf.feature_column.numeric_column(key=key))
        serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
            tf.feature_column.make_parse_example_spec(my_feature_columns))
        estimator_path = model.export_saved_model(output1, serving_input_fn)


def model_predict(input_data, input_model_path, feature_column):
    feature_column = arg_check_transformation("list_name_str", "feature_column", feature_column)
    model_path = get_last_dir(input_model_path)
    imported = tf.saved_model.load(model_path)
    feature_dict = input_data[feature_column]
    # 将输入数据转换成序列化后的 Example 字符串。
    examples = []
    for index, row in feature_dict.iterrows():
        feature = {}
        for col, value in row.iteritems():
            feature[col] = tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
        example = tf.train.Example(
            features=tf.train.Features(
                feature=feature
            )
        )
        examples.append(example.SerializeToString())

    re = imported.signatures["cluster_index"](
        examples=tf.constant(examples))
    return re["output"].numpy()


if __name__ == '__main__':
    tf_k_means_model(feature_column="sepal length (cm),sepal width (cm),petal length (cm),petal width (cm)",
                     center_count=3, input1="./data/iris.csv", output1="./data/")

    data_frame = pd.DataFrame(np.array([[5.0, 3.3, 1.4, 0.2, 0], [7.0, 3.2, 4.7, 1.4, 1]]),
                              columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
                                       'petal width (cm)', 'target'])
    predict = model_predict(
        feature_column="sepal length (cm),sepal width (cm),petal length (cm),petal width (cm)",
        input_model_path="./data/", input_data=data_frame)
    print(predict)
