# _*_ conding:uft-8 _*_
# @Time：2018/7/29 12:30
# @Author：LiHengRui
# @Site：
# @File：线性回归-用电预测.py

"""

有一批描述家庭用电情况的数据，来源于UCI数据库，URL地址：
http://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption
要求：
    使用线性回归对数据进行算法模型预测，并最终得到预测模型（每天各个时间段和功率之间的关
    系、功率与电流之间的关系），分别使用普通最小二乘和sklearn自带的线性回归API来实现
"""

# 导包
import time
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# 设置字符集，防止绘图时出现中文乱码
mpl.rcParams["font.sans-serif"] = ["simHei"]
mpl.rcParams["axes.unicode_minus"] = False

# 使用pands模块的相关API来加载数据(分别为)：
# 日期、时间、有功功率、无功功率、电压、电流、厨房用电功率、洗衣机用电功率、热水器用电功率
path = ".\datas\household_power_consumption_1000.txt"
df = pd.read_csv(filepath_or_buffer=path, sep=";")
print(df.info())    # 打印一下数据信息
"""
-- 数据信息：
RangeIndex: 1000 entries, 0 to 999
Data columns (total 9 columns):
Date(日期)                        1000 non-null object
Time(时间)                        1000 non-null object
Global_active_power(有功功率)     1000 non-null float64
Global_reactive_power(无功功率)   1000 non-null float64
Voltage(电压)                     1000 non-null float64
Global_intensity(电流)            1000 non-null float64
Sub_metering_1(厨房用电功率)      1000 non-null float64
Sub_metering_2(洗衣机用电功率)    1000 non-null float64
Sub_metering_3(热水器用电功率)    1000 non-null float64
dtypes: float64(7), object(2)
memory usage: 70.4+ KB
None
"""

# 异常数据处理：下载的数据集里可能会有异常数据（如缺失值，用“？”代替的地方）
tmp_df = df.replace("?", np.nan)    # 替换缺失值字符为 np.nan
pre_datas = tmp_df.dropna(axis=0, how="any")    # 一行只要有一个数据为np.nan，就删除当前行


# 一. 时间和功率之间的关系
# 数据提取：特征属性（X）为时间，目标属性（Y）为功率值（有功功率）
# 数据信息可知时间是由Date和Time组成，每个字段都是字符串格式，需将其转换为数值型连续变量

# 定义一个函数格式化日期和时间字符串，返回Series格式的的数据
def date_format(date_time):
    # date_time是一个series/tuple；date_time[0]是date，date_time[1]是time
    t = time.strptime("-".join(date_time), "%d/%m/%Y-%H:%M:%S")
    t_date_time = (t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)   # 将日期和时间组合成元组
    return pd.Series(t_date_time)  # 返回Series格式的日期和日间

X_1 = pre_datas.iloc[:, 0:2]
X_1 = X_1.apply(lambda x: date_format(x), axis=1)   # axis = 1（0） 表示按行（列）处理，
# print(type(X_1))
# X_1 = X_1.apply(lambda x: date_format(x), axis=1)
# print(type(X_1.apply(date_format, axis=1)))
# X_1 = X_1.apply(fun, axis=1)
Y_1 = pre_datas.iloc[:, 2]


# 将数据集划分成训练集和测试集
x1_train, x1_test, y1_train, y1_test = train_test_split(X_1, Y_1, train_size=0.8,
                                                    test_size=0.2, random_state=20)
# print(x1_train.shape)
# print(y1_train.shape)
# print(x1_test.shape)
# print(y1_test.shape)

# 1_1. 使用普通最小二乘的方式进行模型训练（这里出现奇异矩阵，无法计算，无法使用此方法）
# x1_train_mat = np.mat(x1_train)
# y1_train_mat = np.mat(y1_train).reshape(-1, 1)
# print(x1_train_mat.shape)
# print(y1_train_mat.shape)
# theta1 = (x1_train_mat.T * x1_train_mat).I * x1_train_mat.T * y1_train_mat
# print(theta1)
# y_predict_1 = np.mat(x1_test) * theta1

# 1_2. 使用sklearn自带的线性回归API进行模型训练
lr_model_1 = LinearRegression(fit_intercept=True) # 构建算法模型对象，训练截距项
lr_model_1.fit(x1_train, y1_train)   # 模型训练
y_predict_1_2 = lr_model_1.predict(x1_test)   # 模型预测

# 打印评估指标信息
print("(时间和功率)训练集上R2:",lr_model_1.score(x1_train, y1_train))
print("(时间和功率)测试集上R2:",lr_model_1.score(x1_test, y1_test))
mse = np.average((y_predict_1_2 - y1_test) ** 2)
rmse = np.sqrt(mse)
print("(时间和功率)rmse:", rmse)

length = np.arange(len(x1_test))
plt.figure(facecolor="w")
plt.plot(length, y_predict_1_2, "r-", linewidth=2, label=u"预测值")
plt.plot(length, y1_test, "g-", linewidth=2, label=u"真实值")
plt.legend(loc="upper right")
plt.title("用电预测(时间和功率)-sklearnAPI")
plt.grid(True)
plt.show()

#######################################################################################################


# 二. 功率和电流之间的关系
X_2 = pre_datas.loc[:, ["Global_active_power", "Global_reactive_power"]]
Y_2 = pre_datas.loc[:, ["Global_intensity"]]
# print(X_2.head(5))
# print(Y_2.head(5))

# 将数据集划分成训练集和测试集
x2_train, x2_test, y2_train, y2_test = train_test_split(X_2, Y_2, train_size=0.8, test_size=0.2, random_state=20)

# 2_1. 使用普通最小二乘的方式进行模型训练
x2_train_mat = np.mat(x2_train)
y2_train_mat = np.mat(y2_train).reshape(-1, 1)

theta2 = (x2_train_mat.T * x2_train_mat).I * x2_train_mat.T * y2_train_mat

# 使用模型参数进行预测
y_predict_2_1 = np.mat(x2_test) * theta2
# 绘图
length = np.arange(len(x2_test))
plt.figure(facecolor="w")
plt.plot(length, y_predict_2_1, "r-", linewidth=2, label=u"预测值")
plt.plot(length, y2_test, "b-", linewidth=2, label=u"真实值")
plt.legend(loc="upper right")
plt.title("用电预测(功率和电流)-最小二乘")
plt.show()

# 2_2. 使用sklearn自带的线性回归API进行模型训练
lr_model_2 = LinearRegression() # 构建算法模型对象，训练截距项
lr_model_2.fit(x2_train, y2_train)  # 模型训练
y_predict_2_2 = lr_model_2.predict(x2_test) # 模型预测

# 打印评估指标信息
print("(功率和电流)训练集上R2:",lr_model_2.score(x2_train, y2_train))
print("(功率和电流)测试集上R2:",lr_model_2.score(x2_test, y2_test))
mse = np.average((y_predict_2_2 - y2_test) ** 2)
rmse = np.sqrt(mse)
print("(功率和电流)rmse:", rmse)
# 绘图
length = np.arange(len(x2_test))
plt.figure(facecolor="w")
plt.plot(length, y_predict_2_2, "r-", linewidth=2, label=u"预测值")
plt.plot(length, y2_test, "b-", linewidth=2, label=u"真实值")
plt.legend(loc="upper right")
plt.title("用电预测(功率和电流)-sklearnAPI")
plt.show()





