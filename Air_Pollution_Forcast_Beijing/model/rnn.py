#!/usr/bin/env python
# encoding:utf-8
import torch
#import torch.utils.data as Data
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset
import torchvision
from torchvision import datasets, transforms
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#from Air_Pollution_Forcast_Beijing.model.data_tranform import scaler, train_x, test_x, train_X, test_X, train_y, test_y
import matplotlib.pyplot as plt
from numpy import concatenate
from math import sqrt
from sklearn.metrics import mean_squared_error
import pickle

import pdb


#https://www.jb51.net/article/139078.htm (详解PyTorch批训练及优化器比较 )

"""基于RNN的分类模型  每一行当做时间特征 """

BATCH_SIZE = 50



def loss_mape(labels, preds, scaler_label=None ):
    if scaler_label:
        labels = labels* scaler_label.data_range_.item() + scaler_label.data_min_.item()
        preds = preds* scaler_label.data_range_.item() + scaler_label.data_min_.item()
    mse = torch.mean(  torch.pow( labels- preds, 2 ) )
    mabe = torch.mean( torch.abs( labels -preds ) )
    mape = torch.mean( torch.abs( (labels-preds)/labels  ) )

    res = (mse, mabe, mape )
    return res



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

    f_file = '../feat/b_w47_s464811_feat.csv'

    df = pd.read_csv( f_file, encoding='utf-8', parse_dates=[0], index_col=0, header=0 )
    print(df.shape)
    print( df.head(3) )

    n_dev = min( (len(df)-1 )*0.1, 10)
    n_train = len(df)-1 - n_dev
    print("ntrain: %s ndev:%s " %(n_train, n_dev))

    scaler_feat = MinMaxScaler(feature_range=(0, 1))
    scaler_label  = MinMaxScaler( feature_range=(1e-4, 1) )
    sca_X = scaler_feat.fit_transform(df.iloc[:,:-1].values )
    sca_y = scaler_label.fit_transform( df.iloc[:,[-1]].values )

    df_nor = pd.concat( [pd.DataFrame(sca_X, index=df.index ), pd.DataFrame(sca_y, index=df.index ) ], axis=1 )
    df = df_nor

    train_X, train_y, dev_X, dev_y = df.iloc[:n_train, :-1], df.iloc[:n_train, [-1]], df.iloc[n_train:-1, :-1], df.iloc[n_train:-1, [-1] ]
    pred_X, pred_y = df.iloc[[-1], :-1], df.iloc[[-1], -1]


    #1、时间序列预测，以前一时刻（t-1）的所有数据预测当前时刻（t）的值

    #X = PM2.5(t-1)  pollution(t-1) ,dew(t-1) ,temp(t-1) ,press(t-1) ,wnd_dir(t-1) ,wnd_spd(t-1) ,snow(t-1) ,rain(t-1)
    #Y = PM2.5(t)
    # 2. 网络搭建
    train_X= torch.tensor(train_X.values, dtype=torch.float32 )
    train_y= torch.tensor(train_y.values, dtype=torch.float32 )
    train_X= train_X.unsqueeze(1)

    test_X= torch.tensor(dev_X.values, dtype=torch.float32 )
    test_y= torch.tensor(dev_y.values, dtype = torch.float32 )
    test_X= test_X.unsqueeze(1)

    pred_X= torch.tensor(pred_X.values, dtype=torch.float32 )
    pred_X= pred_X.unsqueeze(1)

    dtrain = DataLoader( TensorDataset( train_X, train_y ), shuffle=True, batch_size = 16 )
    #ddev = DataLoader( TensorDataset( test_X, test_y ), shuffle=True, batch_size = 16 )

    net=RNN( input_size= train_X.shape[-1], hidden_size=64, output_size=1, num_layers=1, dropout_p=0.05, batch_first=True )

    # 3. 训练
    # 3. 网络的训练（和之前CNN训练的代码基本一样）


    #opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]
    optimizer=torch.optim.Adam(net.parameters(),lr=0.001, weight_decay=0.008 )

    #回归： 最小mse
    #loss_F=torch.nn.MSELoss()
    #交叉熵损失函数
    #loss_F=torch.nn.CrossEntropyLoss()


    mape_train_list = []
    mape_dev_list = []

    best_dev = None
    best_model = None

    for epoch in range(10):
        epoch +=1
        for step , (strain_X, strain_y) in enumerate( dtrain ):
            step+=1
            print(step)
            pred=net(strain_X)
            #break;
            #loss=loss_F(pred,strain_y)
            mse_train, mabe_train, mape_train = loss_mape( strain_y, pred, scaler_label )
            mape_train_list += [mape_train.data.item() ]
            # 计算loss
            optimizer.zero_grad()
            mape_train.backward()
            optimizer.step()
            print("epoch:%s step:%s/%s mse_train: %.3f mabe_train:%.3f mape_train:%.3f" %( epoch, step, strain_X.shape[0],  mse_train.data.item(), mabe_train.data.item(), mape_train.data.item()  ) )
            # 每50步，计算精度
            with torch.no_grad():
                test_pred=net(test_X)
                mse_dev, mabe_dev, mape_dev= loss_mape( test_y, test_pred, scaler_label )
                mape_dev_list += [mape_dev.data.item() ]
                if not best_dev or mape_dev.data.item()< best_dev:
                    best_model = net
                    best_dev = mape_dev.data.item()
                print("epoch:%s step:%s/%s mse_dev: %.3f mabe_dev:%3.f mape_dev:%3.f" %( epoch, step, strain_X.shape[0],  mse_dev.data.item(), mabe_dev.data.item(), mape_dev.data.item()  ) )

    df_loss = pd.DataFrame( np.stack([np.array(mape_train_list), np.array(mape_dev_list) ] , axis=-1 ) ) #, keys=["train_loss", "dev_loss"])
    df_loss.plot()
    plt.show()

    print("best_dev: %.3f" %(best_dev))
    pred = scaler_label.inverse_transform( best_model( pred_X).data ).item()
    print("pred: %.3f " %( pred) )


