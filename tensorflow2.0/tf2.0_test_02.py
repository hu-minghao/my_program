import tensorflow as tf
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras import Model
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
import os
import scipy.io as sio
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow_core.python.keras.callbacks import LearningRateScheduler
import tensorflow_core as tfc
import math
from tensorflow.keras import initializers

tf.__version__


#定义模型
class nn_model(Model):
    def __init__(self):
        super(nn_model,self).__init__()
        self.d1=Dense(64,activation='relu',kernel_initializer=initializers.RandomNormal(stddev=0.05))
        self.d2=Dense(128,activation='relu',kernel_initializer=initializers.RandomNormal(stddev=0.05))
        self.d3=Dense(10,activation='softmax',kernel_initializer=initializers.RandomNormal(stddev=0.05))
        
    def call(self,input):
        x=self.d1(input)
        x=self.d2(x)
        x=self.d3(x)
        return x



#定义学习率衰减函数
def scheduler(epoch):
    init_lr=0.001
    drop=0.6
    step_epoch=10
    lr=init_lr*drop**math.floor((1+epoch)/step_epoch)
    if epoch!=0 and epoch%10==0:
        print('change lr to {}'.format(lr))
    return lr
change_Lr = tfc.python.keras.callbacks.LearningRateScheduler(scheduler)

if __name__=='__main__':
    mat_path = os.path.join('E:/TensorFlow', 'mnist-original.mat')
    mnist = sio.loadmat(mat_path)
    x, y = mnist["data"].T, mnist["label"].T
    #利用sklearn划分模块，划分数据集
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=2020)
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32) / 255.0
    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32) / 255.0
    y_train=tf.cast(y_train,tf.int32)
    y_test=tf.cast(y_test,tf.int32)
    model=nn_model()
    file_path='E:/TensorFlow/model/weight.ckpt'
    cp_callback=tf.keras.callbacks.ModelCheckpoint(file_path,save_weights_only=True,save_best_only=True)
    opt=optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer='adam',loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                 metrics=['sparse_categorical_accuracy'])
    hist=model.fit(x_train,y_train,epochs=50,batch_size=256,validation_data=(x_test,y_test),
                   callbacks=[cp_callback,change_Lr])
    model.summary()
    
    plt.subplot(1,2,1)
    plt.title('Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(hist.history['loss'])
    
    plt.subplot(1,2,2)
    plt.title('Test_Accuray')
    plt.xlabel('Epoch')
    plt.ylabel('Accuray')
    plt.plot(hist.history['val_sparse_categorical_accuracy'])
    plt.show()
