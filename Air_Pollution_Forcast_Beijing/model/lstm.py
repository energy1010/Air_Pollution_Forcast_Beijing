#!usr/bin/env python
# -*- coding: utf-8 -*-

"""
Time:17/1/1
---------------------------
Question:
---------------------------
"""
from keras import Sequential
from keras.layers import LSTM, Dense
from Air_Pollution_Forcast_Beijing.model.data_tranform import scaler, test_x, train_X, test_X, train_y, test_y
import matplotlib.pyplot as plt
from numpy import concatenate  # 数组拼接
from math import sqrt
from sklearn.metrics import mean_squared_error

model = Sequential()
#Input must be three-dimensional, comprised of samples, timesteps, and features.
# specify the input_shape argument that expects a tuple containing the number of timesteps and the number of features
# X = PM2.5(t-1)  pollution(t-1) ,dew(t-1) ,temp(t-1) ,press(t-1) ,wnd_dir(t-1) ,wnd_spd(t-1) ,snow(t-1) ,rain(t-1)
# Y = PM2.5(t)
#(8760, 1, 8) (8760,) (35039, 1, 8) (35039,)
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))

#mae损失
model.compile(loss='mae', optimizer='adam')
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y))

'''
    对数据绘图
'''
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# make the prediction,为了在原始数据的维度上计算损失，需要将数据转化为原来的范围再计算损失
yHat = model.predict(test_X)

'''
    这里注意的是保持拼接后的数组  列数  需要与之前的保持一致
'''
inv_yHat = concatenate((yHat, test_x[:, 1:]), axis=1)   # 数组拼接
#pdb.set_trace()
print(inv_yHat.shape)
inv_yHat = scaler.inverse_transform(inv_yHat)
inv_yHat = inv_yHat[:, 0]

test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_x[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)    # 将标准化的数据转化为原来的范围
inv_y = inv_y[:, 0]

rmse = sqrt(mean_squared_error(inv_yHat, inv_y))
print('Test RMSE: %.3f' % rmse)
