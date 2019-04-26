#!/usr/bin/env python
# encoding:utf-8
import torch
import torch.utils.data as Data
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

from Air_Pollution_Forcast_Beijing.model.data_tranform import scaler, test_x, train_X, test_X, train_y, test_y
import matplotlib.pyplot as plt
from numpy import concatenate  # 数组拼接
from math import sqrt
from sklearn.metrics import mean_squared_error

import pdb

"""基于RNN的分类模型  每一行当做时间特征 """

BATCH_SIZE = 50

class RNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #batch_first: 输入数据的size为[batch_size, time_step, input_size]还是[time_step, batch_size, input_size]
        self.rnn=torch.nn.LSTM( input_size=8, hidden_size=64, num_layers=1, batch_first=True )
        self.out=torch.nn.Linear(in_features=64,out_features=1)

    def forward(self,x):
        # 一下关于shape的注释只针对单项
        # output: [batch_size, time_step, hidden_size]
        # h_n: [num_layers,batch_size, hidden_size]
        # 虽然LSTM的batch_first为True,但是h_n/c_n的第一维还是num_layers
        # c_n: 同h_n
        #pdb.set_trace()
        output, (h_n,c_n)=self.rnn(x)
        #print(output.size())
        # output_in_last_timestep=output[:,-1,:]
        # 也是可以的
        output_in_last_timestep=h_n[-1,:,:] #
        #print(output_in_last_timestep.equal(output[:,-1,:])) #ture
        x=self.out(output_in_last_timestep)
        return x

if __name__ == "__main__":
    # 1. 加载数据
    #dataset_train  = datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)) ]))

    #dataset_test= datasets.MNIST('../data', train=False, transform=transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)) ]))
    #training_dataset = torchvision.datasets.MNIST("./mnist", train=True, transform=torchvision.transforms.ToTensor(), download=True)
    #dataloader = Data.DataLoader(dataset=training_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2) # showSample(dataloader)
    #test_data=torchvision.datasets.MNIST(root="./mnist",train=False, transform=torchvision.transforms.ToTensor(),download=True)
    #test_dataloader=Data.DataLoader( dataset=test_data,batch_size=10000 ,shuffle=False,num_workers=2)
    #dataloader = torch.utils.data.DataLoader( dataset_train, batch_size=BATCH_SIZE, shuffle=True )
    #test_dataloader = torch.utils.data.DataLoader( dataset_test, batch_size=BATCH_SIZE, shuffle=False )


    #testdata_iter=iter(test_dataloader)
    ##test_x [10000, 1, 28, 28]  test_y [10000]
    #test_x,test_y=testdata_iter.next()
    #pdb.set_trace()
    ##batch ,seq, fea
    #test_x=test_x.view(-1,28,28)
    # 2. 网络搭建
    net=RNN()

    # 3. 训练
    # 3. 网络的训练（和之前CNN训练的代码基本一样）

    optimizer=torch.optim.Adam(net.parameters(),lr=0.001)
    #交叉熵损失函数
    loss_F=torch.nn.MSELoss()
    #loss_F=torch.nn.CrossEntropyLoss()

    train_X= torch.tensor(train_X)
    train_y= torch.tensor(train_y)
    train_y = train_y.unsqueeze(1) #[batch, 1]
    train_size = train_X.shape[0]

    #mini-batch
    pdb.set_trace()
    for epoch in range(100):
        pdb.set_trace()
        # 数据集只迭代一次
        for step in range(0, train_size, 50 ):
        #for step, input_data in enumerate(dataloader):
            #x,y=input_data
            strain_X = train_X[step:step+50, : , :]
            strain_y = train_y[step:step+50, : ]
            pred=net(strain_X)
            #break;
            loss=loss_F(pred,strain_y)
            # 计算loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step%2==0:
                print("loss:" + str(loss.item()) )
            # 每50步，计算精度
        #    with torch.no_grad():
        #        test_pred=net(test_x)
                #prob=torch.nn.functional.softmax(test_pred,dim=1)
                #pred_cls=torch.argmax(prob,dim=1)
                #acc=(pred_cls==test_y).sum().numpy()/pred_cls.size()[0]
                #rmse = sqrt(mean_squared_error(, inv_y))
                #print(f"{epoch}-{step}: accuracy:{acc}")


