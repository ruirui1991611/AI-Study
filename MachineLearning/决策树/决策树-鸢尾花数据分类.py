# _*_ conding:uft-8 _*_
# @Time：2018/8/8 22:44
# @Author：LiHengRui
# @File：决策树-鸢尾花数据分类.py


import pandas as pd
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, recall_score

#  设置属性防止绘图时中文乱码
mpl.rcParams["font.sans-serif"] = [u"simHei"]
mpl.rcParams["axes.unicode_minus"] = False

# 1. 加载数据
path = "./datas/iris.data"
# 列名：花萼长度，花萼宽度，花瓣长度，花瓣宽度，类别
column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "category"]
# 由于本数据没有列名，为了观察方便，需要自己在读取数据的时候加上列名
df = pd.read_csv(filepath_or_buffer=path, header=None, names=column_names, sep=",")

# 2. 定义一个函数解析一行数据，将类别转换成相应的数值形式
# 本数据共3个类别：1.Iris-setosa ，2.Iris-versicolor ，3.Iris-virginica
def parse_data(data):
    result = []
    tmp = zip(column_names, data)
    for name, value in tmp:
        if name == "category":
            if value == "Iris-setosa":
                result.append(1)
            elif value == "Iris-versicolor":
                result.append(2)
            else :
                result.append(3)
        else:
            result.append(value)
    return result

# 类别是字符串形式的，无法计算，调用pandas的apply()函数对一行数据进行处理，将类别转换成数值形式
pre_datas = df.apply(lambda x: parse_data(x), axis=1)
X_1 = pre_datas[column_names[0:-1]]
Y_1 = pre_datas[[column_names[-1]]]
# print(X_1)
# print(Y_1)

# 3. 将数据分成训练集个测试集
x_train, x_test, y_train, y_test = train_test_split(X_1, Y_1, train_size=0.9, test_size=0.1, random_state=20)

# 4. 构建决策树算法模型
"""
算法模型API主要参数解释：
# 给定采用gini还是entropy作为纯度的衡量指标
criterion="gini", 
# 进行划分特征选择的时候采用什么方式来选择，best表示每次选择的划分特征都是全局最优的(所有特征属性中的最优划分)；random表示每次选择的划分特征不是所有特征属性中的最优特征，而且先从所有特征中随机的抽取出部分特征属性，然后在这个部分特征属性中选择最优的，也就是random选择的是局部最优。
# best每次都选择最优的划分特征，但是这个最优划分特征其实是在训练集数据上的这一个最优划分。但是这个最优在实际的数据中有可能该属性就不是最优的啦，所以容易陷入过拟合的情况 --> 如果存在过拟合，可以考虑使用random的方式来选择。
splitter="best", 
# 指定构建的决策树允许的最高层次是多少，默认不限制
max_depth=None,
# 指定进行数据划分的时候，当前节点中包含的数据至少要去的数据量
min_samples_split=2,
min_samples_leaf=1,
min_weight_fraction_leaf=0.,
# 在random的划分过程中，给定每次选择局部最优划分特征的时候，使用多少个特征属性
max_features=None,
random_state=None,
max_leaf_nodes=None,
min_impurity_split=1e-7,
class_weight=None,
presort=False
"""
decision_tree_algo = DecisionTreeClassifier(criterion="gini")

# 5. 算法模型的训练
decision_tree_algo.fit(x_train, y_train)

# 6. 模型效果评估
# 准确率 ：P = TP/(TP+FP)
print("训练集上的精确率：{}".format(decision_tree_algo.score(x_train, y_train)))
print("测试集上的精确率：{}".format(decision_tree_algo.score(x_test, y_test)))
# 召回率：R = TP/(TP+FN)
print("训练集上的召回率：{}".format(recall_score(y_train, decision_tree_algo.predict(x_train), average="macro")))
print("测试集上的召回率：{}".format(recall_score(y_test, decision_tree_algo.predict(x_test), average="macro")))
# F1值(准确率和召回率的调和均值)：F1 = 2PR/(P+R)
print("训练集上的F1值：{}".format(f1_score(y_train, decision_tree_algo.predict(x_train), average="macro")))
print("测试集上的F1值：{}".format(f1_score(y_test, decision_tree_algo.predict(x_test), average="macro")))

# 7. 可视化输出
# 使用pydotplus插件生成png文件和pdf文件（使用pip install安装模块）
from sklearn import tree
import pydotplus

# 主要参数解释
# feature_names=None, class_names=None 分别给定特征属性和目标属性的name信息
# filled=True 填充颜色
dot_data = tree.export_graphviz(decision_tree=decision_tree_algo, out_file=None,
                                feature_names=['sepal length', 'sepal width', 'petal length', 'petal width'],
                                class_names=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], filled=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png("iris2.png")
graph.write_pdf("iris2.pdf")
