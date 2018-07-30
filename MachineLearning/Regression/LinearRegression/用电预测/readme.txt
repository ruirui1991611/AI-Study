datas目录下是描述个人家庭用电情况的数据，来源于UCI数据库：http://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption

注：household_power_consumption.txt文件是从UCI上下载的原始数据，但是其中有很多的缺失值(即"?"处的值)，而且数据量比较大(有120多M，这里将其压缩成.zip文件)，为了学习方便，从原始数据中截取了部分组成新的
数据：
	-- household_power_consumption_1000.txt文件里有1000条数据
	
	
需求：
	使用线性回归对数据进行算法模型预测，并最终得到预测模型（1.每天各个时间段和功率之间的关系，2.功率与电流之间的关系），分别使用普通最小二乘和sklearn自带的线性回归API来实现
	
