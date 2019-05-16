#!/usr/bin/env python
# encoding:utf-8
import torch
import torch.utils.data as Data
import torchvision
from torchvision import datasets, transforms
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Air_Pollution_Forcast_Beijing.model.data_tranform import scaler, train_x, test_x, train_X, test_X, train_y, test_y
import matplotlib.pyplot as plt
from numpy import concatenate  # 数组拼接
from math import sqrt
from sklearn.metrics import mean_squared_error

import pdb


#https://www.jb51.net/article/139078.htm (详解PyTorch批训练及优化器比较 )

"""基于RNN的分类模型  每一行当做时间特征 """

BATCH_SIZE = 50

class RNN(torch.nn.Module):

    def __init__(self, input_size=8, hidden_size=64, output_size=1, num_layers=1, dropout_p=0.05, batch_first=True ):
        super().__init__()
        #batch_first: 输入数据的size为[batch_size, time_step, input_size]还是[time_step, batch_size, input_size]
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.batch_first = batch_first


        self.rnn=torch.nn.LSTM( input_size=self.input_size, hidden_size= self.hidden_size, num_layers= self.num_layers, batch_first= self.batch_first)
        self.dropout = torch.nn.Dropout(self.dropout_p)
        self.out=torch.nn.Linear(self.hidden_size,  self.output_size )

    def forward(self,x):
        # 一下关于shape的注释只针对单项
        # output: [batch_size, time_step, hidden_size]
        # h_n:    [num_layers,  batch_size,  hidden_size]
        # c_n:    [num_layers, batch_size, hidden_size]
        # 虽然LSTM的batch_first为True,但是h_n/c_n的第一维还是num_layers
        # c_n: 同h_n
        #pdb.set_trace()
        output, (h_n,c_n)=self.rnn(x)
        #print(output.size())
        # output_in_last_timestep=output[:,-1,:]
        # 也是可以的

        #取最后的输出结果做回归
        output_in_last_timestep=h_n[-1,:,:] #
        #print(output_in_last_timestep.equal(output[:,-1,:])) #ture
        #回归
        x=self.out(output_in_last_timestep)
        return x

if __name__ == "__main__":

    print("main")
    # 1. 加载数据

    #1、时间序列预测，以前一时刻（t-1）的所有数据预测当前时刻（t）的值

    #X = PM2.5(t-1)  pollution(t-1) ,dew(t-1) ,temp(t-1) ,press(t-1) ,wnd_dir(t-1) ,wnd_spd(t-1) ,snow(t-1) ,rain(t-1)
    #Y = PM2.5(t)
    #pdb.set_trace()
    ##batch ,seq, fea
    #test_x=test_x.view(-1,28,28)
    # 2. 网络搭建

    net=RNN( input_size=8, hidden_size=64, output_size=1, num_layers=1, dropout_p=0.05, batch_first=True )

    # 3. 训练
    # 3. 网络的训练（和之前CNN训练的代码基本一样）


    #opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]
    optimizer=torch.optim.Adam(net.parameters(),lr=0.001)

    #回归： 最小mse
    loss_F=torch.nn.MSELoss()
    #交叉熵损失函数
    #loss_F=torch.nn.CrossEntropyLoss()

    #pdb.set_trace()
    train_X= torch.tensor(train_X)
    train_y= torch.tensor(train_y)
    train_y = train_y.unsqueeze(1) #[batch, 1]
    train_size = train_X.shape[0]

    test_X= torch.tensor(test_X)
    test_y= torch.tensor(test_y)
    test_y = test_y.unsqueeze(1) #[batch, 1]
    test_size = test_X.shape[0]
    #mini-batch
    #pdb.set_trace()
    #for epoch in range(100):
    #    pdb.set_trace()
    #    # 数据集只迭代一次
    #    for step in range(0, train_size, 50 ):
    #    #for step, input_data in enumerate(dataloader):
    #        #x,y=input_data
    #        print(step)
    #        if step+50>train_size:
    #            strain_X = train_X[step:train_size, : , :]
    #            strain_y = train_y[step:train_size, : ]
    #        else:
    #            strain_X = train_X[step:step+50, : , :]
    #            strain_y = train_y[step:step+50, : ]
    #        pred=net(strain_X)
    #        #break;
    #        loss=loss_F(pred,strain_y)
    #        # 计算loss
    #        optimizer.zero_grad()
    #        loss.backward()
    #        optimizer.step()
    #        if step%2==0:
    #            print("loss:" + str(loss.item()) )
    #        # 每50步，计算精度
    #    #    with torch.no_grad():
    #    #        test_pred=net(test_x)
    #            #prob=torch.nn.functional.softmax(test_pred,dim=1)
    #            #pred_cls=torch.argmax(prob,dim=1)
    #            #acc=(pred_cls==test_y).sum().numpy()/pred_cls.size()[0]
    #            #rmse = sqrt(mean_squared_error(, inv_y))
    #            #print(f"{epoch}-{step}: accuracy:{acc}")

    #全局统计RMSE
    for epoch in range(1000):
        # 数据集只迭代一次
        #for step, input_data in enumerate(dataloader):
            #x,y=input_data
        pred=net(train_X)
        #pdb.set_trace()
        #break;
        loss=loss_F(pred, train_y)



        # 计算loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch%2==0:
            print("train ori loss:" + str(loss.item()) )

            #pdb.set_trace()
            pred_arr = torch.tensor(pred).detach().numpy()
            train_arr = torch.tensor(train_x[:, 1:]).detach().numpy()
            inv_pred= np.concatenate( ( pred_arr, train_arr ) , axis =1)
            #print(inv_pred_test.shape)
            inv_pred= scaler.inverse_transform( inv_pred)
            inv_pred= inv_pred[:,0]

            inv_train_y = np.concatenate( ( train_y,  train_arr ), axis=1 )
            inv_train_y = scaler.inverse_transform( inv_train_y)
            inv_train_y = inv_train_y[:, 0]
            loss1 = mean_squared_error( inv_train_y, inv_pred )
            print("train loss1: " + str( loss1 ) )
            loss2 = loss_F( torch.tensor( inv_pred), torch.tensor(inv_train_y)  )
            print("train loss:" + str(loss2.item()) )




        with torch.no_grad():
            #pdb.set_trace()
            # 数据集只迭代一次
            pred_test = net(test_X) # [batch,1]
            loss=loss_F(pred_test, test_y)
            #rmse = sqrt(mean_squared_error(, inv_y))
            #print(f"{epoch}-{step}: accuracy:{acc}")
            print("test ori loss:" + str(loss.item()) )
            #pdb.set_trace()
            inv_pred_test = np.concatenate( ( pred_test, test_x[:, 1:] ) , axis =1)
            #print(inv_pred_test.shape)
            inv_pred_test = scaler.inverse_transform( inv_pred_test )
            inv_pred_test = inv_pred_test[:,0]

            #test_y = test_y.reshape((len(test_y), 1))
            inv_test_y = concatenate((test_y, test_x[:, 1:]), axis=1)
            inv_test_y = scaler.inverse_transform(inv_test_y)    # 将标准化的数据转化为原来的范围
            inv_test_y = inv_test_y[:, 0]
            loss=loss_F( torch.tensor( inv_pred_test), torch.tensor( inv_test_y) )
            print("test loss:" + str(loss.item()) )
