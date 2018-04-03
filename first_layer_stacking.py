import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from heamy.dataset import Dataset
from heamy.estimator import Regressor
from heamy.pipeline import ModelsPipeline
from sklearn import cross_validation
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor,\
    AdaBoostRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR
from tqdm import tqdm
from catboost import CatBoostRegressor
from xgboost import plot_importance
import xgboost as xgb
import warnings
import scipy
import time
import csv
start_time = time.time()


# 警告信息不显示在终端
def ignore_warn(*args, **kwargs):
    pass


warnings.warn = ignore_warn
# 读取数据
data = pd.read_csv('./ShiTou_new.csv')
# print(np.shape(data))       # (44639, 45)
# 计算各特征的缺失率
na_count = data.isnull().sum().sort_values(ascending=False)
na_rate = na_count / len(data)
# 缺失值的特征和缺失率
na_data = pd.concat([na_count, na_rate], axis=1, keys=['count', 'ratio'])
# print("##############输出缺失率最大的前三十各数据的特征和缺失率############")
# print(na_data.head(30))
'''
          count  ratio
n05          44639    1.0
c31          44639    1.0
c15          44639    1.0
c16          44639    1.0
c17          44639    1.0
c18          44639    1.0
c12          44639    1.0
c11          44639    1.0
c09          44639    1.0
n04          44639    1.0
c08          44639    1.0
c07          44639    1.0
c13          44639    1.0
c30          44639    1.0
Unnamed: 28  44639    1.0
c32          44639    1.0
c36          44639    1.0
Unnamed: 42  44639    1.0
c40          44639    1.0
c39          44639    1.0
c38          44639    1.0
c33          44639    1.0
c37          44639    1.0
c35          44639    1.0
c34          44639    1.0
c14          44639    1.0
c10              0    0.0
c06              0    0.0
c05              0    0.0
c03              0    0.0
             count  ratio
n05          44639    1.0
c31          44639    1.0
c15          44639    1.0
c16          44639    1.0
c17          44639    1.0
c18          44639    1.0
c12          44639    1.0
c11          44639    1.0
c09          44639    1.0
n04          44639    1.0
c08          44639    1.0
c07          44639    1.0
c13          44639    1.0
c30          44639    1.0
Unnamed: 28  44639    1.0
c32          44639    1.0
c36          44639    1.0
Unnamed: 42  44639    1.0
c40          44639    1.0
c39          44639    1.0
c38          44639    1.0
c33          44639    1.0
c37          44639    1.0
c35          44639    1.0
c34          44639    1.0
c14          44639    1.0
c10              0    0.0
c06              0    0.0
c05              0    0.0
c03              0    0.0
'''
# 删除缺失率为100%的数据
data = data.drop(na_data[na_data['count'] > 44638].index, axis=1)
# 删除完全缺失的数据后的数据
# print(np.shape(data))       # (44639, 19)
# 删除特征为零的数据
data = data.drop(['c22', 'n01', 'n02', 'c23', 'n03'], axis=1)
data = data.loc[:, :]
data = data[data['c04'] >= 0.0]
# 对缺失值进行填充
index_Nan_c25 = list(data['c25'][data['c25'].isnull()].index)
for i in index_Nan_c25:
    c25_mean = data['c25'].median()
    c25_pred = data['c25'][((data['c24'] == data.iloc[i]['c24']) & (data['c05'] == data.iloc[i]['c05']) &
                           (data['c10'] == data.iloc[i]['c10']))].median()
    if not np.isnan(c25_pred):
        data['c25'].iloc[i] = c25_pred
    else:
        data['c25'].iloc[i] = c25_mean
# c04&c21之间画出散点图观察离群点，对离群点进行删除
'''
# ['c01', 'c02', 'c03', 'c05', 'c06', 'c10', 'c24', 'c25', 'c04']
data_scatter = pd.concat([data['c04'], data['c25']], axis=1)
data_scatter.plot.scatter(x='c25', y='c04', ylim=(0, 40))
plt.title("c25&c04 distribution")
plt.show()
'''
# 删除离群值较大的点
data = data[(data['c03'] > 6.9) & (data['c05'] < 15000) & (data['c24'] < 170)]
# 划分特征和标签
data_y = data['c04']
data_drop = pd.concat([data['c26'], data['c27']], axis=1)
data_drop = pd.concat([data_drop, data['c28']], axis=1)
data_XX = data.drop(['c26', 'c27', 'c28', 'c19', 'c21', 'c04'], axis=1)
'''
data_y2 = data['c27']
data_y = pd.concat([data_y1, data_y2], axis=1)
data_y3 = data['c28']
data_y = pd.concat([data_y, data_y3], axis=1)
'''
# print("###############获取特征名称##################")
# 获取特征名称
quantity = [attr for attr in data_XX.columns if data_XX.dtypes[attr] != 'object']
sns.distplot(data['c04'])
print("Skewness: %f" % data['c04'].skew())
print("Kurtosis: %f" % data['c04'].kurt())
data['c04'] = np.log1p(data['c04'])
sns.distplot(data['c04'])
# print(quantity)
# ['c01', 'c02', 'c03', 'c05', 'c06', 'c10', 'c19', 'n03', 'c21', 'c24', 'c25']
quantity_feature = np.asarray(quantity)
# print("####################对缺失值进行填充############")
'''
# print(data_y)
# data_x = data_x.fillna(0)       # 缺失值填充0
# data_x = data_x.fillna(method='pad')    # 前向填充
'''
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(data_XX)
# print("################对输入数据进行标准化#############")
'''
# 训练数据归一化
scaler = StandardScaler()
data_X = scaler.fit_transform(data_X)
# print(np.shape(data_X))       # (44639, 11)
data_y = scaler.fit_transform(data_y)
# print("################计算特征重要性##################")
feature_important = RandomForestRegressor()
# print(feature_important)
# 查看RandomForestClassifier参数列表
feature_important.fit(data_X, data_y)
importances = feature_important.feature_importances_
# print(importances)
std = np.std([tree.feature_importances_ for tree in feature_important.estimators_], axis=0)
indices = np.argsort(importances)[::-1]
# print(indices)
print("Feature ranking")
for f in range(data_X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
plt.figure()
plt.title("Feature importances")
plt.bar(range(data_X.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
plt.xticks(range(data_X.shape[1]), quantity_feature[indices])
plt.xlim([-1, data_X.shape[1]])
plt.show()
'''
# print("###############特征重要性分布表################")
'''
Feature ranking
1. feature 7 (0.527160)
2. feature 5 (0.124523)
3. feature 2 (0.113162)
4. feature 9 (0.108148)
5. feature 4 (0.042494)
6. feature 10 (0.041936)
7. feature 3 (0.018448)
8. feature 1 (0.011100)
9. feature 0 (0.010443)
10. feature 6 (0.001583)
11. feature 8 (0.001003)
'''
# print("############对原始数据进行降维操作,将处理后的九维数据降为六维#############")
# print("#########使用PCA进行降维操作##########")
'''
pca = PCA(n_components=6)
data_X = pca.fit_transform(data_X)
data_y1 = pca.fit_transform(data_y1)
'''
# print("##########使用LDA进行降维操作#########")
'''
LDA = LinearDiscriminantAnalysis(n_components=6)
data_X = LDA.fit_transform(data_X)
data_y1 = LDA.fit_transform(data_y1)
'''
# print("#########将dataframe类型数据转化为array类型########")
data_X = np.asarray(data_XX)
data_y = np.asarray(data_y)
data_y = data_y.reshape(-1, 1)
# 对数进行标准化操作
scaler = StandardScaler()
data_X = scaler.fit_transform(data_X)
data_y = scaler.fit_transform(data_y)
data_X = pd.DataFrame(data_X)
data_y = pd.DataFrame(data_y)
# 训练集，测试集划分
# train_X, test_X, train_y, test_y = train_test_split(data_X, data_y1, test_size=0.8, random_state=7)
# print(np.shape(train_X))
# print("################模型融合##############")
x_train, x_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.33, random_state=2018)
y_train = np.asarray(y_train).reshape(-1, 1)
# 创建数据集
dataset = Dataset(x_train, y_train.ravel(), x_test)
# 创建RF模型和LR模型
model_lr = Regressor(dataset=dataset, estimator=LinearRegression, parameters={'normalize': True}, name='lr')
model_rf = Regressor(dataset=dataset, estimator=RandomForestRegressor, parameters={'n_estimators': 50}, name='rf')
model_gbdt = Regressor(dataset=dataset, estimator=GradientBoostingRegressor, parameters={'n_estimators': 50, 'learning_rate': 0.05,
                         'max_depth': 4, 'max_features': 'sqrt', 'min_samples_leaf': 15, 'min_samples_split': 10}, name='gbdt')
model_xgb = Regressor(dataset=dataset, estimator=xgb.XGBRegressor, parameters={'n_estimators': 50, 'learning_rate': 0.05,
                            'max_depth': 3}, name='xgb')

# stack两个模型
pipeline = ModelsPipeline(model_lr, model_rf, model_gbdt, model_xgb)
stack_ds = pipeline.stack(k=10, seed=111)
# 第二层使用xgboost模型stack
stacker = Regressor(dataset=stack_ds, estimator=xgb.XGBRegressor)
results = stacker.predict()
# 使用10折交叉验证结果
results10 = stacker.validate(k=10, scorer=mean_squared_error)
print("r2_score: %f" % r2_score(y_test, results))

test_y = pd.DataFrame(y_test)
predictions = pd.DataFrame(results)
data = pd.concat([data_XX, data_drop], axis=1)
data = pd.concat([data, data_y], axis=1)
data = pd.concat([data, predictions], axis=1)
data = np.array(data)
with open('C:/20180402_pre_test.csv', 'w') as f:
    header = ['c01', 'c02', 'c03', 'c05', 'c06', 'c10', 'c24', 'c25', 'c26', 'c27', 'c28', 'c04', 'pred']
    writer = csv.writer(f, delimiter=",")
    writer.writerow(header)
    m = len(data)
    for i in range(m):
        writer.writerow(data[i])
standand_test_y = test_y * test_y.std() + test_y.mean()
standand_predictions = predictions * predictions.std() + predictions.mean()
# predictions.to_csv('C:/20180327_pre.csv')
predictions = np.asarray(standand_predictions)
test_y = np.asarray(standand_test_y)
# print("################使用matplotlib画图展示拟合效果###########")
plt.figure()
plt.plot(predictions[500:700, ], color="r", linewidth=2, linestyle="-", label="predict_DO")
plt.plot(test_y[500:700, ], color="y", linewidth=2, linestyle="-", label="True_DO")
plt.title("stacking_Model")
plt.legend(loc='best')
plt.xlabel("test data")
plt.ylabel("test result")
plt.show()
total_time = time.time() - start_time
print("运行花费总时间: %f" % total_time)