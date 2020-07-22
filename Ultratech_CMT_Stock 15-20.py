import math
import pandas as pd
import numpy as np
from bsedata.bse import BSE
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM,Dense
from sklearn.preprocessing import MinMaxScaler
import pandas_datareader as web


df = web.DataReader('ULTRACEMCO.NS', data_source = 'yahoo', start = '2015-01-01', end = '2020-01-01')

df = df.filter(['Close'])

plt.figure(figsize = (40, 10))
plt.title('UltraTech Cement Limited (ULTRACEMCO.NS) Closing_Price 15-20', fontdict = {'fontsize' : '25'})
plt.xlabel('Date', fontdict = {'fontsize' : '18'})
plt.ylabel('Price in INR (Rs)', fontdict = {'fontsize' : '18'})
plt.plot(df)

dataset = df.values

scaler = MinMaxScaler(feature_range = (0, 1))
scaled_data = scaler.fit_transform(dataset)

train_data_len = math.ceil(len(dataset) * 0.8)

train_data = scaled_data[:train_data_len, :]
x_train = list();y_train = list()

for i in range(80, len(train_data)):
    x_train.append(train_data[i - 80: i, 0])
    y_train.append(train_data[i , 0])
    
x_train = np.array(x_train);y_train = np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential()
model.add(LSTM(100, return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(LSTM(100, return_sequences = False))
model.add(Dense(1024, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1))

model.compile(optimizer = 'adam',
              loss = 'mean_squared_error',
              metrics = ['accuracy'])

hist = model.fit(x_train,y_train, epochs = 50, validation_split = 0.1)

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Val'])
plt.show()

test_data = list()
test_data = scaled_data[train_data_len - 80 : , :]
x_test = list();y_test = list()

for i in range(80, len(test_data)):
    x_test.append(test_data[i- 80: i, 0])
    y_test.append(test_data[i, 0])
    
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicitions = model.predict(x_test)
predicitions = scaler.inverse_transform(predicitions)
predicitions.shape
training_data = pd.DataFrame();testing_data = pd.DataFrame()

model.evaluate(x_test, y_test)

training_data['Close'] = df['Close'][0:train_data_len]
testing_data['Close'] = df['Close'][train_data_len: ]
testing_data['Predicited Closing_Price'] = 0
testing_data['Predicited Closing_Price'] = predicitions

plt.figure(figsize = (40, 10))
plt.title('UltraTech Cement Limited (ULTRACEMCO.NS) Closing_Price 15-20', fontdict = {'fontsize' : '25'})
plt.xlabel('Date', fontdict = {'fontsize' : '18'})
plt.ylabel('Price in INR (Rs)', fontdict = {'fontsize' : '18'})
plt.plot(training_data)
plt.plot(testing_data[['Close', 'Predicited Closing_Price']])
plt.legend(['Training Data', 'Testing Data', 'AI_Model_Predicted Price'])
print(testing_data)





















































































