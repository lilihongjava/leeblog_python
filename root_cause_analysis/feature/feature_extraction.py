# encoding: utf-8
"""
@author: lee
@time: 2020/6/8 15:07
@file: feature_extraction.py
@desc: 
"""
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import gc

from sklearn.feature_extraction.text import TfidfVectorizer


# 告警标题tfidf
def alarm_title_tfidf(data):
    feat = data[['涉及告警基站或小区名称', '告警标题']].copy()
    feat = feat.groupby(['涉及告警基站或小区名称'])['告警标题'].agg(lambda x: ' '.join(x)).reset_index(
        name='Product_list')  # 涉及告警基站或小区名称分组，每组标题join一起
    tf_idf = TfidfVectorizer(max_features=100)
    tf_vec = tf_idf.fit_transform(feat['Product_list'].values.tolist())  # 提取关键词
    tf_df = pd.DataFrame(tf_vec.toarray())
    tf_df['涉及告警基站或小区名称'] = feat['涉及告警基站或小区名称'].values
    tf_df.columns = ['idf_' + str(i + 1) for i in range(tf_df.shape[1] - 1)] + ['涉及告警基站或小区名称']
    return tf_df


# 告警统计特征
def alarm_statistics_feature(data):
    # pandas.Period.XX
    data['告警发生时间_mon'] = data['告警发生时间'].dt.month  # 获取时间的月份部分
    data['告警发生时间_day'] = data['告警发生时间'].dt.day  # 获取时间发生月份中的一天。
    data['告警发生时间_dayofyear'] = data['告警发生时间'].dt.dayofyear  # 获取时间在一年中哪一天，1到365
    data['告警发生时间_hour'] = data['告警发生时间'].dt.hour  # 获取时间的小时部分
    data['告警发生时间_weekday'] = data['告警发生时间'].dt.weekday  # 获取该时间所在星期几，星期一=0，星期日=6
    data['告警发生时间_wy'] = data['告警发生时间'].dt.weekofyear  # 获取时间在一年当中的第几周
    data['告警发生时间_是否周末'] = data['告警发生时间_weekday'].apply(lambda x: 1 if x >= 5 else 0)
    data['告警发生时间_int'] = data['告警发生时间'].apply(lambda x: x.value // 10 ** 9)
    data['告警发生时间_diff'] = data.groupby(['涉及告警基站或小区名称'])['告警发生时间_int'].diff()
    return data


# 告警标题按周w2v
def alarm_title_w2v(data):
    data = data.sort_values('告警发生时间')
    feat = data.groupby(['涉及告警基站或小区名称', '告警发生时间_wy'])['告警标题'].apply(list).reset_index()
    feat = feat.sample(frac=1.0, random_state=112)
    sen = feat['告警标题'].values.tolist()  # 数组里一个标题当中一个句子
    try:
        fastmodel = Word2Vec.load('model_fusaigaojing.txt')
    except:
        fastmodel = Word2Vec(sen, size=100, window=8, min_count=1, workers=4, iter=10)
        fastmodel.save('model_fusaigaojing.txt')
    w2v = []
    for i in range(len(sen)):
        w2v.append(np.mean(fastmodel.wv[sen[i]], axis=0))  # 得到词对应的向量
    w2v_df = pd.DataFrame(w2v)
    w2v_df.columns = ['告警标题_' + str(i + 1) for i in w2v_df.columns]
    w2v_df['涉及告警基站或小区名称'] = feat['涉及告警基站或小区名称'].values
    w2v_df['告警发生时间_wy'] = feat['告警发生时间_wy'].values
    w2v_df = w2v_df.rename(columns={'告警发生时间_wy': '故障发生时间_wy'})
    return w2v_df


# 故障统计特征
def failure_statistics_feature(label):
    label['故障发生时间_mon'] = label['故障发生时间'].dt.month
    label['故障发生时间_day'] = label['故障发生时间'].dt.day
    label['故障发生时间_dayofyear'] = label['故障发生时间'].dt.dayofyear
    label['故障发生时间_hour'] = label['故障发生时间'].dt.hour
    label['故障发生时间_weekday'] = label['故障发生时间'].dt.weekday  # 该时间段所在星期几
    label['故障发生时间_wy'] = label['故障发生时间'].dt.weekofyear  # 计算一年当中的第几周
    label['故障发生时间_是否周末'] = label['故障发生时间_weekday'].apply(lambda x: 1 if x >= 5 else 0)

    label['故障发生时间_day_count'] = label.groupby(['涉及告警基站或小区名称', '故障发生时间_dayofyear'])['工单编号'].transform('count')
    label['故障发生时间_week_count'] = label.groupby(['涉及告警基站或小区名称', '故障发生时间_wy'])['工单编号'].transform('count')
    label['故障发生时间_hour_count'] = label.groupby(['涉及告警基站或小区名称', '故障发生时间_hour'])['工单编号'].transform('count')
    label['故障发生时间_周末_count'] = label.groupby(['涉及告警基站或小区名称', '故障发生时间_是否周末'])['工单编号'].transform('count')
    label['故障发生时间_weekday_count'] = label.groupby(['涉及告警基站或小区名称', '故障发生时间_weekday'])['工单编号'].transform('count')
    label['故障发生时间_故障发生时间_count'] = label.groupby(['涉及告警基站或小区名称', '故障发生时间'])['工单编号'].transform('count')

    label['小区故障时间_min'] = label.groupby(['涉及告警基站或小区名称'])['故障发生时间_dayofyear'].transform('min')
    label['小区故障时间_max'] = label.groupby(['涉及告警基站或小区名称'])['故障发生时间_dayofyear'].transform('max')
    label['小区故障时间_ptp'] = label.groupby(['涉及告警基站或小区名称'])['故障发生时间_dayofyear'].transform(np.ptp)

    label['故障发生时间_int'] = label['故障发生时间'].apply(lambda x: x.value // 10 ** 9)
    label['故障发生时间_diff'] = label.groupby(['涉及告警基站或小区名称'])['故障发生时间_int'].diff()
    label['故障发生时间_diff_min'] = label.groupby(['涉及告警基站或小区名称'])['故障发生时间_diff'].transform('min')
    label['故障发生时间_diff_max'] = label.groupby(['涉及告警基站或小区名称'])['故障发生时间_diff'].transform('max')
    label['故障发生时间_diff_mean'] = label.groupby(['涉及告警基站或小区名称'])['故障发生时间_diff'].transform('mean')
    label['故障发生时间_diff_std'] = label.groupby(['涉及告警基站或小区名称'])['故障发生时间_diff'].transform('std')
    return label


def alarm_title_features(label, data):
    # 告警标题统计特征，分组统计'_dayofyear', '_wy', '_hour', '_是否周末', '_weekday'告警标题个数
    for i in ['_dayofyear', '_wy', '_hour', '_是否周末', '_weekday']:
        alarm_happen = data.groupby(['涉及告警基站或小区名称', '告警发生时间' + i])['告警标题'].count().reset_index(
            name='告警发生时间' + i + '_count')
        alarm_happen.columns = ['涉及告警基站或小区名称', '故障发生时间' + i, '告警发生时间' + i + '_count']
        label = label.merge(alarm_happen, on=['涉及告警基站或小区名称', '故障发生时间' + i], how='left')
    # 计算每个唯一值的个数
    for i in ['_dayofyear', '_wy', '_hour', '_是否周末', '_weekday']:
        alarm_happen = data.groupby(['涉及告警基站或小区名称', '告警发生时间' + i])['告警标题'].nunique().reset_index(
            name='告警发生时间' + i + '_nunique')
        alarm_happen.columns = ['涉及告警基站或小区名称', '故障发生时间' + i, '告警发生时间' + i + '_nunique']
        label = label.merge(alarm_happen, on=['涉及告警基站或小区名称', '故障发生时间' + i], how='left')
    return label


# groupby涉及告警基站或小区名称
def make_feature(data, aggs, name):
    agg_data = data.groupby('涉及告警基站或小区名称').agg(aggs)
    agg_data.columns = agg_data.columns = ['_'.join(i).strip() + name for i in agg_data.columns.tolist()]
    return agg_data.reset_index()
