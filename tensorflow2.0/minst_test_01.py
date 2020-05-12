import scipy.io as sio
import os
import  tensorflow as tf
from    tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
tf.__version__ #'2.0.0-rc1'
#下载好的数据集，通过sio模块读取
mat_path = os.path.join('E:/TensorFlow', 'mnist-original.mat')
mnist = sio.loadmat(mat_path)
x, y = mnist["data"].T, mnist["label"].T

x.shape
#数据归一化
x=x/255.0

#利用sklearn划分模块，划分数据集
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=2020)
x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)
y_train=tf.cast(y_train,tf.int32)
y_test=tf.cast(y_test,tf.int32)
# 构建dataset对象，批处理
train_db = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(1000)
test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(1000)
 
# 构建模型中会用到的权重,使用截断正态分布
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1,seed=2020))
b1 = tf.Variable(tf.random.truncated_normal([1,256], stddev=0.1,seed=2020))
w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1,seed=2020))
b2 = tf.Variable(tf.random.truncated_normal([1,128], stddev=0.1,seed=2020))
w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1,seed=2020))
b3 = tf.Variable(tf.random.truncated_normal([1,10], stddev=0.1,seed=2020))
 

lr = 0.05
train_loss=[]
test_acc=[]
epoch=5
loss_all=0

for epoch in range(epoch):
    for step,(x_train,y_train) in enumerate(train_db):
        with tf.GradientTape() as tape:
            # 一层影藏层模型搭建
            h1 = tf.matmul(x_train,w1) + b1
            h1 = tf.nn.relu(h1)
            h2 = tf.matmul(h1,w2) + b2
            h2 = tf.nn.relu(h2)
            out = tf.matmul(h2,w3) + b3
            out=tf.nn.softmax(out)
            # 把标签转化成one_hot编码 
            x_test = tf.cast(x_test, tf.float32)
            y_ = tf.one_hot(y_train, depth=10)
            #计算损失
            loss=tf.reduce_mean(tf.square(out-y_))
            loss_all+=loss.numpy()
        grads=tape.gradient(loss,[w1, b1, w2, b2, w3, b3])
        #参数更新
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])
        #5epoch，btach=1000
        print('Epoch{},loss:{}'.format(epoch,loss_all/49))
        #记录loss
        train_loss.append(loss_all/49)
        loss_all=0
        
        total_correct,total_number=0,0
        
        for x_test,y_test in test_db:
            #前项传播计算预测值
            h1 = tf.matmul(x_test,w1) + b1
            h1 = tf.nn.relu(h1)
            h2 = tf.matmul(h1,w2) + b2
            h2 = tf.nn.relu(h2)
            out = tf.matmul(h2,w3) + b3
            out=tf.nn.softmax(out)
            pred=tf.argmax(out,axis=1)
            #计算正确率
            pred=tf.cast(pred,dtype=y_test.dtype)
            correct=tf.cast(tf.equal(pred,y_test),dtype=tf.int32)
            correct=tf.reduce_sum(correct)
            total_correct+=int(correct)
            total_number+=x_test.shape[0]
        
        acc=total_correct/total_number
        test_acc.append(acc)
        print('test acc:',acc)
        print('-'*100)

plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(train_loss)
plt.legend()
plt.show()


plt.title('Acc Curve')
plt.xlabel('Epoch')
plt.ylabel('acc')
plt.plot(test_acc)
plt.legend()
plt.show()  
            
            
