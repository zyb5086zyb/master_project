import codecs
import math
import random
import numpy as np
from math import tanh
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, Normalizer
import csv

random.seed(0)


def rand(a, b):  # 随机函数
    return (b - a) * random.random() + a


def make_matrix(m, n, fill=0.0):  # 创建一个指定大小的矩阵
    mat = []
    for i in range(m):
        mat.append([fill] * n)      # 构造成m行n列的0矩阵
    return mat

# 定义sigmoid函数和它的导数
def softsign(x):
    return 1.0/(1 + abs(x))

def softsign_derivate(x):
    return softsign(x)*softsign(x)

def sigmoid (x) :
    return 1.0 / (1 + math.exp(-1.0*x))


def sigmoid_derivate(x):
    return sigmoid(x) * (1.0 - sigmoid(x))  # sigmoid函数的导数

def Tanh(x):

    return tanh(x)

def Tanh_derivate(y):
    return 1-Tanh(y)*Tanh(y)

# 将数据归一化到(-1, 1)内
def MaxMin(train):
    for i in range(np.shape(train)[0]):
        for j in range(np.shape(train)[1]):
            train[i][j] = (2 * train[i][j] - max(train[j]) - min(train[j]))/(max(train[j] - min(train[j])))
    return train

class BPNeuralNetwork:
    def __init__(self):  # 初始化变量
        self.input_n = 0        # 输入层神经元节点数目
        self.hidden_n = 0       # 隐层神经元节点数目
        self.output_n = 0       # 输出层神经元数目
        self.hidden_value = []      # 隐层结点的阈值
        self.output_value = []      # 输出层节点的阈值
        self.input_cells = []       # 输入层各神经元结点的输出值(节点值)
        self.hidden_cells = []      # 隐层各神经元结点的输出值(节点值)
        self.output_cells = []      # 输出层各神经元结点的输出值(节点值)
        self.input_weights = []     # 输入层与隐层各结点之间的连接权值
        self.output_weights = []    # 隐层与输出层各节点之间的连接权值
        self.input_correction = []      # 输入层各神经元结点的纠正矩阵
        self.output_correction = []     # 输出层各神经元结点的纠正矩阵
        # 三个列表维护：输入层，隐含层，输出层神经元

    def setup(self, ni, nh, no):
        self.input_n = ni + 1   # 输入层+偏置项
        self.hidden_n = nh      # 隐含层
        self.output_n = no      # 输出层

        # 初始化神经元
        self.input_cells = [1.0] * self.input_n
        self.hidden_cells = [1.0] * self.hidden_n
        self.output_cells = [1.0] * self.output_n
        self.hidden_value = [1.0] * self.hidden_n
        self.output_value = [1.0] * self.output_n

        # 初始化连接边的边权
        self.input_weights = make_matrix(self.input_n, self.hidden_n)  # 邻接矩阵存边权：输入层->隐藏层
        self.output_weights = make_matrix(self.hidden_n, self.output_n)  # 邻接矩阵存边权：隐藏层->输出层

        # 随机初始化边权：为了反向传导做准备--->随机初始化的目的是使对称失效
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                self.input_weights[i][h] = rand(-0.2, 0.2)  # 由输入层第i个元素到隐藏层第j个元素的边权为随机值

        for h in range(self.hidden_n):
            for o in range(self.output_n):
                self.output_weights[h][o] = rand(-2.0, 2.0)  # 由隐藏层第i个元素到输出层第j个元素的边权为随机值

        # 保存校正矩阵，为了以后误差做调整
        self.input_correction = make_matrix(self.input_n, self.hidden_n)
        self.output_correction = make_matrix(self.hidden_n, self.output_n)

        # 输出预测值

    def predict(self, inputs, label):      # 对样例的预测
        # 对输入层进行操作转化样本
        for i in range(self.input_n - 1):       # 输入层添加的偏置结点去除
            self.input_cells[i] = inputs[i]     # n个样本从0~n-1

        # 计算隐藏层的输出，每个节点最终的输出值就是{[(输入层和输出层的连接权值)*输入层的节点值]的加权和}
        for j in range(self.hidden_n):
            total = 0.0
            for i in range(self.input_n):
                total += self.input_cells[i] * self.input_weights[i][j]
                # 此处为何是先i再j，以隐含层节点做大循环，输入样本为小循环，是为了每一个隐藏节点计算一个输出值，传输到下一层
            self.hidden_value[j] = total
            self.hidden_cells[j] = Tanh(self.hidden_value[j])  # 此节点的输入是前一层所有输入点和到该点之间的权值加权和
        # 计算输出层的输出，每个节点的最终的输出值就是{[(隐层和输出层的连接权值)*隐层的节点值]得加权和}
        for k in range(self.output_n):
            total = 0.0
            for j in range(self.hidden_n):
                total += self.hidden_cells[j] * self.output_weights[j][k]       # 输出层的预测值
            self.output_value[k] = total
            self.output_cells[k] = Tanh(self.output_value[k] - label[k])  # 输出层的预测值与标签值之间差值的激励函数

        return self.output_cells[:]   # 最后输出层的结果返回

    # 反向传播算法：调用预测函数，根据反向传播获取权重后前向预测，将结果与实际结果返回比较误差
    def back_propagate(self, case, label, learn, correct):
        # 对输入样本做预测
        self.predict(case, label)  # 对实例进行预测,获取预测的输出层的节点值
        output_deltas = [0.0] * self.output_n  # 初始化矩阵
        for o in range(self.output_n):
            error = label[o] - self.output_cells[o]  # 正确结果和预测结果的误差：0,1，-1
            output_deltas[o] = Tanh_derivate(self.output_cells[o]) * error  # 误差稳定在0~1内
            # 输出层的误差率，输出层的预测值的导数与误差的乘积

        # 隐含层误差
        hidden_deltas = [0.0] * self.hidden_n
        for h in range(self.hidden_n):
            error = 0.0
            for o in range(self.output_n):
                error += output_deltas[o] * self.output_weights[h][o]   # 输出层误差率与(隐层和输出层神经元连接权值)差值
            hidden_deltas[h] = Tanh_derivate(self.hidden_cells[h]) * error       # 隐层误差率
            # 反向传播算法求W
        # 更新隐藏层->输出权重
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                change = output_deltas[o] * self.hidden_cells[h]        # 输出层的变化值：输出层的误差值与隐层的节点值的乘积
                # 调整权重：原本权重值+上一层每个节点的权重学习*变化+矫正率*纠正值
                self.output_weights[h][o] += learn * change + correct * self.output_correction[h][o]
                self.output_correction[h][o] = change

        # 更新输入->隐藏层的权重
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                change = hidden_deltas[h] * self.input_cells[i]         # 隐层的变化值：隐层的预测值与输入层的节点值得乘积
                # 调整权重：原本权重值+上一层每一个结点的权重学习*变化+矫正率*纠正值
                self.input_weights[i][h] += learn * change + correct * self.input_correction[i][h]
                # 输入层的纠正值更新
                self.input_correction[i][h] = change

        # 获取全局误差
        error = 0.0
        for o in range(len(label)):
            error += preprocessing.normalize(0.5 * (label[o] - self.output_cells[o]) * (label[o] - self.output_cells[o]))     # 输出层每一个节点的真实输出值与预测结点值的差值的平方的一半
            # print(error)
        return error

    def train(self, cases, labels, limit=100, learn=0.01, correct=0.1):
        # cases输入层的节点值 labels输出层的真实值 limit总的迭代次数 learn学习率 correct纠正率
        for i in range(limit):  # 设置迭代次数
            error = 0.0
            for j in range(len(cases)):  # 对输入层进行访问
                label = labels[j]
                case = cases[j]
                error += self.back_propagate(case, label, learn, correct)  # 样例，标签，学习率，正确阈值


    def test(self):  # 学习异或
        filename = r'./ShiTou_new2.csv'
        df = pd.read_csv(filename)           # 0~35行数据不读取 skiprows=[36]
        # df = df.dropna(how='all')   # how='all' 表示删除一行数据全部为NaN axis=1 表示删除一列数据全部为NaN
        # df = df.drop(['c04', 'c22', 'n01', 'n02', 'c23'], axis=1)   # 删除列号为'c04','c22','n02'
        # df = df.duplicated()     # 判断重复数据
        # df = df.drop_duplicates(['c26', 'c27', 'c28'])    # 删除重复数据
        # 切分所有的输入数据

        num = 30000    # 对测试和训练数据进行切分
        num1 = 40000
        row = 7
        train_x = df.iloc[:num, :row].as_matrix().astype(float)    # astype 函数将数据转换为指定类型)
        train_y = df.iloc[:num, row:10].as_matrix().astype(float)
        test_x = df.iloc[num:num1, :row].as_matrix().astype(float)
        test_y = df.iloc[num:num1, row:10].as_matrix().astype(float)


        # 对数据进行预处理进行归一化
        # train_x = MaxMin(train_x)
        # train_y = MaxMin(train_y)
        # test_x = MaxMin(test_x)
        # test_y = MaxMin(test_y)
        # train_x, test_x, train_y, test_y = train_test_split(train_x, train_y)
        scaler = MinMaxScaler(feature_range=(0, 1))         # 最小最大值标准化[0-1]

        # 对训练数据进行归一化
        train_x = scaler.fit_transform(train_x)
        train_y = scaler.fit_transform(train_y)
        # print(train_y)
        # 对测试数据进行归一化
        test_x = scaler.fit_transform(test_x)
        test_y = scaler.fit_transform(test_y)
        # 训练数据进行正则化
        '''
        train_x_normalizer = Normalizer().fit(train_x)
        train_x = train_x_normalizer.transform(train_x)
        train_y_normalizer = Normalizer().fit(train_y)
        train_y = train_y_normalizer.transform(train_y)
        '''
        #  对测试数据进行正则
        test_x_normalize = Normalizer().fit(test_x)
        test_x = test_x_normalize.transform(test_x)
        test_y_normalizer = Normalizer().fit(test_y)
        test_y = test_y_normalizer.transform(test_y)
        self.setup(7, 6, 3)  # 初始化神经网络：输入层，隐藏层，输出层元素个数
        self.train(train_x, train_y, 1, 0.01, 0.1)  # 可以更改
        test_result = make_matrix(np.shape(test_y)[0], np.shape(test_y)[1])
        for item in range(np.shape(test_x)[0]):
            test_data = test_x[item]
            test = test_y[item]
            test_result[item] = self.predict(test_data, test)
            print(test_y[item], test_result[item])
        test_result = np.array(test_result)
        test_result = abs(test_result)
        with open('C:/csvFile4.csv', 'w') as g:
            header = ['COD', 'NH4N', 'TP']
            writer = csv.writer(g, delimiter=",")
            writer.writerow(header)
            m = len(test_result)
            for i in range(m):
                writer.writerow(test_result[i])
        # 一张图片上显示五个结果
        plt.figure(1)
        plt.plot(test_result[:, 0:1], color="r", linewidth=2, linestyle="-", label="Predict_COD")
        plt.plot(test_result[:, 1:2], color="orange", linewidth=1, linestyle="-", label="Predict_NH4N")
        plt.plot(test_result[:, 2:3], color="g", linewidth=1, linestyle="-", label="Predict_TP")
        plt.legend(loc='best')
        plt.title("Predict")
        plt.xlabel("test data")
        plt.ylabel("test result")
        # 输入数据与预测数据对比图

        ax = plt.figure(2)
        ax.add_subplot(3, 1, 1)
        plt.plot(test_result[:, 0:1], color="r", linewidth=2, linestyle="-", label="predict_COD")
        plt.plot(test_y[:, 0:1], color="y", linewidth=2, linestyle="-", label="True_COD")
        plt.title("Predict_True_COD")
        plt.legend(loc='best')
        plt.xlabel("test data")
        plt.ylabel("test result")
        ax.add_subplot(3, 1, 2)
        plt.plot(test_result[:, 1:2], color="r", linewidth=2, linestyle="-", label="Predict_NH4N")
        plt.plot(test_y[:, 1:2], color="y", linewidth=2, linestyle="-", label="True_NH4N")
        plt.legend(loc='best')
        plt.xlabel("test data")
        plt.ylabel("test result")
        ax.add_subplot(3, 1, 3)
        plt.plot(test_result[:, 2:3], color="r", linewidth=2, linestyle="-", label="Predict_TP")
        plt.plot(test_y[:, 2:3], color="y", linewidth=2, linestyle="-", label="True_TP")
        plt.legend(loc='best')
        plt.xlabel("test data")
        plt.ylabel("test result")
        plt.show()


if __name__ == '__main__':
    nn = BPNeuralNetwork()
    nn.test()