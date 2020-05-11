import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

#读取数据
file_dir='E:\\GDBT_LR\\loan\\'
train_data='gbdt_train.csv'
test_data='gdbt_test.csv'
train=pd.read_csv(file_dir+train_data)
test=pd.read_csv(file_dir+test_data)
#删除无用参数
del train['Unnamed: 0']
del test['Unnamed: 0']

#取数据集
data=train[data_list]
test_data=test[data_list]

#构造训练集和测试集
feature=[x for x in data_list if x!='loan_status']
X_train=data[feature]
y_train=data['loan_status']
X_test=test_data[feature]
y_test=test_data['loan_status']


# 构造lgb分类器
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 64,
    'num_trees': 100,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

# 设置叶子节点
num_leaf = 64

print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=100,
                valid_sets=lgb_train)

print('Save model...')
# save model to file
gbm.save_model('model.txt')

print('Start predicting...')
# predict and get data on leaves, training data
y_pred = gbm.predict(X_train, pred_leaf=True)
print(np.array(y_pred).shape)
print(y_pred[0])

#样本个数行，树个数*叶子树列矩阵
transformed_training_matrix = np.zeros([len(y_pred), len(y_pred[0]) * num_leaf],
                                       dtype=np.int64)  # N * num_tress * num_leafs
#将转换矩阵按叶子树划分，将叶子预测的节点位置添加标记，标记位置为temp数组，在大矩阵中，在相应位置处的元素加一
for i in range(0, len(y_pred)):
    temp = np.arange(len(y_pred[0])) * num_leaf + np.array(y_pred[i])
    transformed_training_matrix[i][temp] += 1
    
#预测集做同样的处理
y_pred = gbm.predict(X_test, pred_leaf=True)
print('Writing transformed testing data')
transformed_testing_matrix = np.zeros([len(y_pred), len(y_pred[0]) * num_leaf], dtype=np.int64)
for i in range(0, len(y_pred)):
    temp = np.arange(len(y_pred[0])) * num_leaf + np.array(y_pred[i])
    transformed_testing_matrix[i][temp] += 1
    
lm = LogisticRegression(penalty='l2',C=0.05) # logestic model construction
lm.fit(transformed_training_matrix,y_train)  # fitting the data
y_pred_test = lm.predict_proba(transformed_testing_matrix)   # Give the probabilty on each label

print(y_pred_test)

NE = (-1) / len(y_pred_test) * sum(((1+y_test)/2 * np.log(y_pred_test[:,1]) +  (1-y_test)/2 * np.log(1 - y_pred_test[:,1])))
print("Normalized Cross Entropy " + str(NE))

#以阀值为0.5看查全率与查准率
def get_pcr(y_tar,y_pre):
    id_list=[]
    for i in range(len(y_tar)):
        if y_tar[i]==1:
            id_list.append(i)
    right_n=0
    for i in id_list:
        if y_pre[i][0]<y_pre[i][1]:
            right_n+=1
    pre_id=[]
    for i in range(len(y_pre)):
        if y_pre[i][0]<y_pre[i][1]:
            pre_id.append(i)
    good_pre=set(pre_id)&set(id_list)
    print('查准率为：{}'.format(len(good_pre)/len(pre_id)))
    print('查全率为：{}'.format(right_n/len(id_list)))
    
get_pcr(y_test,y_pred_test)#查准率为：0.9205776173285198，查全率为：0.6623376623376623

y_pred_train = lm.predict_proba(transformed_training_matrix)   # Give the probabilty on each label
get_pcr(y_train,y_pred_train)#查准率为：0.9971139971139971，查全率为：0.9262734584450402
