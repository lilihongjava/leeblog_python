# encoding: utf-8
"""
@author: lee
@time: 2020/7/20 16:11
@file: text_util.py
@desc: 
"""
import re

import jieba


def stopwords_list():
    return [line.strip() for line in open('./data/hit_stopwords.txt', encoding='UTF-8').readlines()]


# 对句子进行中文分词
def seg_depart(sentence, stopwords):
    sentence_depart = jieba.cut(sentence.strip())
    out_str = ''
    # 去停用词
    for word in sentence_depart:
        # keep English, digital and Chinese ^A-Z^a-z^0-9^\u4e00-\u9fa5
        if word not in stopwords and not re.match(r'[^A-Z^a-z^0-9^\u4e00-\u9fa5]', word):
            out_str += word
            out_str += " "
    return out_str
