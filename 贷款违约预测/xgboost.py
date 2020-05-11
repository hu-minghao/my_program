import xgboost as xgb
from sklearn.metrics import accuracy_score

#测试下xgboost效果
dtrain=xgb.DMatrix(X_train,y_train)
dtest=xgb.DMatrix(X_test,y_test)
param={'max_depth':5,'learning_rate':0.1, 'n_estimators':100, 'verbosity':1, 
        'objective':'binary:logistic'}
xgb_model = xgb.train(param, dtrain, num_boost_round=30)
pre=xgb_model.predict(dtest)
pre0=[round(x) for x in pre]
accuracy0=accuracy_score(pre0,y_test)

print('TestAccuracy:{}'.format(accuracy0))

#xgb查全率  0.612987012987013
xgb_list=[]
for i in range(len(pre0)):
    if pre0[i]==1:
        xgb_list.append(i)
len(set(xgb_list)&set(id_list))/len(id_list)

#xgb查准率  0.9957805907172996
len(set(xgb_list)&set(id_list))/len(xgb_list)
