# coding=utf-8
import os


# 判断目录是否存在
def path_exists(path):
    if not os.path.isdir(path):
        raise Exception("目录 %s 不存在!" % path)
    return path


def get_file(file_dir):
    # 返回目录下第一个文件
    try:
        files = os.listdir(file_dir)
        file_path = os.path.join(file_dir, files[0])
    except Exception:
        raise Exception("%s获取文件出错" % file_dir)
    return file_path


def get_last_dir(file_dir):
    # 返回最新目录
    try:
        lists = os.listdir(file_dir)
        lists.sort(key=lambda fn: os.path.getmtime(os.path.join(file_dir, fn)), reverse=True)
    except Exception:
        raise Exception("%s获取目录出错" % file_dir)
    return os.path.join(file_dir, lists[0])
