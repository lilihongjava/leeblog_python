# encoding: utf-8
"""
@author: lee
@time: 2020/6/8 9:38
@file: main.py
@desc: 
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing


from root_cause_analysis.common.common import cc_label
from root_cause_analysis.feature.feature_extraction import make_feature, alarm_title_tfidf, alarm_statistics_feature, \
    alarm_title_w2v, failure_statistics_feature, alarm_title_features
from root_cause_analysis.model.train import xgb_train

if __name__ == '__main__':
    # 读取数据
    path = './data/'
    label = pd.read_csv(open(path + 'training_order.csv'))
    data = pd.read_csv(open(path + 'train_alarm.csv'))
    # 数据预处理
    label = label.drop_duplicates(['故障发生时间', '涉及告警基站或小区名称', '故障原因定位（大类）']).reset_index(drop=True)  # 去重
    label['故障发生时间'] = pd.to_datetime(label['故障发生时间'])
    label = label.sort_values(by=['涉及告警基站或小区名称', '故障发生时间']).reset_index(drop=True)
    # 故障原因定位（大类）转换为数字
    label['故障原因定位（大类）'] = label['故障原因定位（大类）'].map(cc_label)

    data = data[~data['涉及告警基站或小区名称'].isnull()]
    data = data.drop_duplicates().reset_index(drop=True)
    data['告警发生时间'] = data['告警发生时间'].apply(lambda x: x.replace('FEB', '02'))
    data['告警发生时间'] = data['告警发生时间'].apply(lambda x: x.replace('MAR', '03'))
    data['告警发生时间'] = pd.to_datetime(data['告警发生时间'], format="%d-%m-%Y %H:%M:%S")

    # 特征工程
    # 针对告警标题做tfidf
    tf_df = alarm_title_tfidf(data)
    label = label.merge(tf_df, on='涉及告警基站或小区名称', how='left')

    # 告警历史统计特征
    data = alarm_statistics_feature(data)
    aggs = {
        '告警标题': ['count', 'nunique'],
        '告警发生时间_是否周末': ['mean'],
        '告警发生时间_hour': ['nunique'],
        '告警发生时间_weekday': ['nunique'],
        '告警发生时间_day': ['nunique'],
        '告警发生时间_wy': ['nunique'],
        '告警发生时间_dayofyear': ['nunique', 'max', 'min', np.ptp],
        '告警发生时间_diff': ['min', 'max', 'mean', 'std'],
    }
    agg_df = make_feature(data, aggs, "_告警")  # groupby小区名称
    label = label.merge(agg_df, on='涉及告警基站或小区名称', how='left')

    # 故障历史统计特征
    label = failure_statistics_feature(label)
    aggs = {
        '故障发生时间_是否周末': ['mean'],
        '故障发生时间_hour': ['nunique'],
        '故障发生时间_weekday': ['nunique'],
        '故障发生时间_day': ['nunique'],
        '故障发生时间_wy': ['nunique'],
        '故障发生时间_dayofyear': ['nunique', 'max', 'min', np.ptp]
    }
    agg_df = make_feature(label, aggs, "_故障")
    label = label.merge(agg_df, on='涉及告警基站或小区名称', how='left')
    # 告警标题统计特征
    label = alarm_title_features(label, data)

    # 涉及告警基站或小区名称标签化
    label['涉及告警基站或小区名称_code'] = preprocessing.LabelEncoder().fit_transform(label['涉及告警基站或小区名称'])
    label['涉及告警基站或小区名称_count'] = label.groupby('涉及告警基站或小区名称')['故障发生时间'].transform('count')

    # 告警标题按周w2v，按同小区，故障发生时间_wy(时间在一年当中的第几周),告警发生时间_wy合并
    w2v_df = alarm_title_w2v(data)
    label = label.merge(w2v_df, on=['涉及告警基站或小区名称', '故障发生时间_wy'], how='left')

    # 小时展开表特征 0-23
    pivot = pd.pivot_table(label, index='涉及告警基站或小区名称', columns='故障发生时间_hour', values=['工单编号'],
                           aggfunc='count').reset_index().fillna(0)  # pivot_table 透视表，值为工单编号，提取每个小区小时(0~23)区间工单数
    pivot.columns = ['涉及告警基站或小区名称'] + ['gu' + str(int(i)) for i in pivot['工单编号'].columns.tolist()]
    label = label.merge(pivot, on='涉及告警基站或小区名称', how='left')
    # 比率特征
    for i in ['告警标题_count_告警', '告警标题_nunique_告警']:
        label[i + '_ratio'] = label['涉及告警基站或小区名称_count'] / (label[i] + 1)
    for i in ['hour_nunique', 'weekday_nunique', 'day_nunique', 'wy_nunique', 'dayofyear_nunique']:  # 按同小区
        label[i + '_ratio'] = label['故障发生时间_' + i + '_故障'] / (label['告警发生时间_' + i + '_告警'] + 1)
    # 涉及告警基站或小区名称、故障发生时间分组，然后rank时间
    label['rank'] = label.groupby('涉及告警基站或小区名称')['故障发生时间'].rank(method='dense')

    print(label)
    print(label.shape)

    train = label[~label['故障原因定位（大类）'].isnull()]

    # 过滤字段
    col = [i for i in train.columns if i not in ['涉及告警基站或小区名称', '故障发生时间',
                                                 '工单编号', '告警发生时间',
                                                 '故障原因定位（大类）']]
    X_train = train[col].copy().reset_index(drop=True)
    y_train = train['故障原因定位（大类）'].copy().reset_index(drop=True).astype(int)

    print(X_train.shape, y_train.shape)

    # 模型训练
    f1_list, logsocre, accuracy_list = xgb_train(X_train, y_train)
    print("mean_accuracy=", np.mean(accuracy_list), np.std(accuracy_list))
    print("mean_f1=", np.mean(f1_list), np.std(f1_list))
    print("mean_mlog=", np.mean(logsocre), np.std(logsocre))
