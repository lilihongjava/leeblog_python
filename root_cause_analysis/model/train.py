# encoding: utf-8
"""
@author: lee
@time: 2020/6/17 10:55
@file: train.py
@desc: 
"""
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold


def xgb_train(X_train, y_train):
    K = 5
    seed = 2021
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)  # 分层采样，确保训练集，测试集中各类别样本的比例与原始数据集中相同
    f1_list = []
    logsocre = []
    accuracy_list = []

    for i, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[train_index, :].copy(), X_train.iloc[test_index, :].copy()
        y_tr, y_val = y_train.iloc[train_index].copy(), y_train.iloc[test_index].copy()
        print("Fold ", i + 1)

        xgb_tr = xgb.DMatrix(X_tr, y_tr)  # 训练
        xgb_val = xgb.DMatrix(X_val, y_val)  # 验证
        xgb_params = {"objective": 'multi:softprob',
                      "booster": "gbtree",
                      "eta": 0.07,
                      "max_depth": 9,
                      "subsample": 0.85,
                      'eval_metric': 'mlogloss',
                      'num_class': 23,
                      "colsample_bylevel": 0.7,
                      # 'tree_method': 'gpu_hist', gpu
                      'lambda': 6,
                      "thread": 12,
                      "seed": 666
                      }
        watchlist = [(xgb_tr, 'train'), (xgb_val, 'eval')]
        xgb_model = xgb.train(xgb_params,
                              xgb_tr,
                              num_boost_round=2666,  # 提升迭代的次数，也就是生成多少基模型
                              evals=watchlist,  # 用于对训练过程中进行评估列表中的元素
                              verbose_eval=200,  # 要求evals 里至少有一个元素。如果输入数字，假设为5，则每隔5个迭代输出一次
                              early_stopping_rounds=100)  # 早期停止次数 ，假设为100，验证集的误差迭代到一定程度在100次内不能再继续降低，就停止迭代。

        pred = xgb_model.predict(xgb_val, ntree_limit=xgb_model.best_ntree_limit)
        # argmax返回的是最大数的索引，即哪个大类的可能性最大
        val_pred = [np.argmax(x) for x in pred]
        accuracy = accuracy_score(y_val, val_pred)  # 准确率
        logsocre.append(xgb_model.best_score)
        print("accuracy_score=", accuracy)
        accuracy_list.append(accuracy)
        f1 = f1_score(y_val, val_pred, average='weighted')  # f1
        print("f1=", accuracy)
        f1_list.append(f1)

    return f1_list, logsocre, accuracy_list
