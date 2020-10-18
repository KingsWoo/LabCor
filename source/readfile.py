# coding:utf-8

import csv
import numpy as np

def read_csv(path, title=True):
    """
    直接读取带标签行的csv文件
    :param path:
    :return: 标签列表和数据集
    """
    with open(path, 'r') as csvfile:
        
        reader = csv.reader(csvfile)
        mat = [row for row in reader]
        
        if(title):
            title = mat[0]
            mat.remove(title)
        else:
            title = []

        print('reading file %s;' % path)

        return np.array(mat, dtype=float), title


def write_csv(path, data, mode = 'w'):
    """
    write matrix into csv file
    :param path:
    :param data:
    :return:
    """
    with open(path, mode, newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in data:
            writer.writerow(row)

