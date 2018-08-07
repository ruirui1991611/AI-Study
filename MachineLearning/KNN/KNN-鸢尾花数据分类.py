#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/7 11:14
# @Author  : LiHengRui
# @File    : KNN-鸢尾花数据分类.py
# @Software: PyCharm

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV    # 网格交叉验证模块，用于选取K(邻居数)值等参数
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score    # F1评估指标模块
from sklearn.metrics import recall_score    # 召回率评估指标模块

# 1. 加载数据
file_path = "./datas/iris.data"
# 列名：花萼长度，花萼宽度，花瓣长度，花瓣宽度，类别
column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "category"]
# 由于本数据没有列名，为了观察方便，需要自己在读取数据的时候加上列名
df = pd.read_csv(filepath_or_buffer=file_path, header=None, names=column_names, sep=",")


# 2. 定义一个函数解析一行数据，将类别转换成相应的数值形式
# 本数据共3个类别：1.Iris-setosa ，2.Iris-versicolor ，3.Iris-virginica
def parse_data(data):
    result = []
    tmp = zip(column_names, data)   # zip()将可迭代的对象中对应的元素打包成元组，并返回由这些元组组成的列表
    for name, value in tmp:
        if name == "category":
            if value == "Iris-setosa":
                result.append(1)
            elif value == "Iris-versicolor":
                result.append(2)
            else:
                result.append(3)
        else:
            result.append(value)
    return result

# 由于类别是字符串形式的，无法计算，调用pandas的apply()函数对一行数据进行处理，将类别转换成数值形式
pre_datas = df.apply(lambda x: parse_data(x), axis=1)
print(pre_datas.head(10))
print(pre_datas.category.value_counts())

X = pre_datas.iloc[0:, 0:-1]
Y = pre_datas.iloc[0:, -1]
# X = pre_datas[column_names[0:-1]]
# Y = pre_datas[column_names[-1]]

# 3. 将数据分成训练集个测试集
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.9, test_size=0.1, random_state=20)


# 4. 构建KNN算法模型，KNeighborsClassifier()关键参数解释如下：
# n_neighbors: 邻居数，默认为5; 一般需要调整的参数
# weights: 做预测的时候采用何种预测方式，是等权重还是不同权重(常用 - uniform：等权重，distance：与距离成反比)
# algorithm：寻找K近邻时采用的方式(如brute或kd_tree)
# leaf_size: 构建Tree的过程中，停止构建的条件，最多的叶子数目
# p&metric: 样本相似度度量方式，默认为欧氏距离，p = 2
# (API参考：http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
knn_algo = KNeighborsClassifier()

# 使用 sklearn的GridSearchCV() 函数进行KNN算法模型的参数调优，函数的关键参数解释如下：
# estimator: 给定算法模型对象(这里为：knn_algo)
# param_grid: 给定estimator对应的算法模型可选的参数列表有哪些，是一个dict字典数据类型
# param_grid中的key是estimator算法模型对应的参数名,是字符串形式；value是该参数可选的值列表集合
param_grid = {
    "n_neighbors": [2, 5, 10],
    "weights": ["uniform", "distance"],
    "algorithm": ["kd_tree", "brute"],
    "leaf_size": [5, 10, 20]
}
knn_final_algo = GridSearchCV(estimator=knn_algo, param_grid=param_grid, cv=5)	# cv=5，代表5折验证

# 5. 算法模型的训练
knn_final_algo.fit(x_train, y_train)

# 获取实际最优模型
print("最优模型:{}".format(knn_final_algo.best_estimator_))
print("最优模型对应的参数:{}".format(knn_final_algo.best_params_))

# 6. 模型效果评估
# 准确率 ：P = TP/(TP+FP)
print("训练集上的准确率:{}".format(knn_final_algo.best_estimator_.score(x_train, y_train)))
print("测试集上的准确率:{}".format(knn_final_algo.best_estimator_.score(x_test, y_test)))
# 召回率：R = TP/(TP+FN)
print("训练集上的召回率:{}".format(recall_score(y_train, knn_final_algo.predict(x_train), average="macro")))
print("测试集上的召回率:{}".format(recall_score(y_test, knn_final_algo.predict(x_test), average="macro")))
# F1值(准确率和召回率的调和均值)：F1 = 2PR/(P+R)
print("训练集上的F1值:{}".format(f1_score(y_train, knn_final_algo.predict(x_train), average="macro")))
print("测试集上的F1值:{}".format(f1_score(y_test, knn_final_algo.predict(x_test), average="macro")))

