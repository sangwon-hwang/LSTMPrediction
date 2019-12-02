
# coding: utf-8

# Data Preprocess
# Load raw data, modify and write files which have a format for gradient Hawkes
import pandas
import datetime
import time

# data load 1 out of 11 from original data
# download file http://labs.criteo.com/wp content/uploads/2014/07/criteoconversionlogs.tar.gz
# in terminal run tar -xvzf //haloaround.tistory.com/25
# in terminal run split -l 1000000
# change name xxa to data_a.txt 
dir = "./data_a.txt" # path
file = open(dir, 'r', encoding='utf-8')

# user-defined function: transfer each string to integer
def trans_format(temp): 
    temp_int_list = []
    for tt in temp:
        temp_int = int(tt)
        temp_int_list.append(temp_int)

    return temp_int_list

file.seek(1)
lines = file.readlines()
lines_list = []

for line in lines:
    temp_line = line.replace("\t",",").split(',')
    # \t -> comma 
    # split(): string -> list 
    lines_list.append(temp_line) 
    
# set up DataFrame to load a list of lists

header = ['click timestamp',
		  'conversion timestamp',
		  'IF1',
		  'IF2',
		  'IF3',
		  'IF4',
		  'IF5',
		  'IF6',
		  'IF7',
		  'IF8',
		  'CF1',
		  'CF2',
		  'CF3',
		  'CF4',
		  'CF5',
		  'CF6',
		  'CF7',
		  'CF8',
		  'CF9']
df_lines = pandas.DataFrame(lines_list, columns = header) 

# Data Extract
tr_nums = int(len(df_lines))  # 1000000 elements
df_lines_tr = df_lines[0:tr_nums]
cl_times = df_lines_tr[df_lines_tr['click timestamp'] != ''] # df_lines_tr is a list which contains lists of strings
cl_times_train = trans_format(cl_times['click timestamp'])   # type: int 

# milli sec config
import random
cl_times_train = list(map(lambda x : x+round(random.random(), 4) , cl_times_train))
cl_times_train.sort()

# pandas dataFrame 
final_cl_times_train_df = pandas.DataFrame({'time':cl_times_train, 'magnitude':1})
# set index
final_cl_times_train_df = final_cl_times_train_df.set_index("time")

import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# create_dataset
# modification : -look_back-1 -> -look_back
def create_dataset(dataset, look_back):
	dataX, dataY = [], []
	for i in range(len(dataset) - look_back):
		a = dataset[i:(i + look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset
cl_dataframe = final_cl_times_train_df.time[(final_cl_times_train_df.index >= 355800 ) & (final_cl_times_train_df.index < 363000)]

look_back = 1

cl_dataset = cl_dataframe.values[733040:736302] # period to predict in click

### MOD Leaning, Prediction and Evaluation ###

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(cl_dataset)

# MODIFIED : split into train and test sets
train_size = 2836 
test_size = len(dataset) - train_size
                                     # test_size = test_size - look_back
                                     # cl_dataset[train_size-lookback:len(dataset),:]
train, test = dataset[0:train_size,:], dataset[train_size-look_back:len(dataset),:]

# MODIFIED : reshape into X=t and Y=t+1
look_back = 2
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=31, batch_size=1, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

### Leaning, Prediction and Evaluation ###
# 13 RMSE where lookbak window is 2 and epoch is equal to 20 
# 116.34 RMSE where lookbak window is 2 and epoch is equal to 10 
# 8.02 RMSE where lookbak window is 2 and epoch is equal to 25 
# 7.25 RMSE where lookbak window is 2 and epoch is equal to 30 

d2 = numpy.r_[trainPredict, testPredict]
final_cl_times_predict_df = pandas.DataFrame(d2)
final_cl_times_predict_df.to_csv("filename.csv", header="time", index=True)
