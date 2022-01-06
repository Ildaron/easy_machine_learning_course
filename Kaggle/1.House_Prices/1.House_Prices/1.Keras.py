import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras_visualizer import visualizer 

#1. Data prepairing
data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')
data_train=data_train.fillna('0')

data_train_col=data_train.values
col_len=(len(data_train_col[0]))
count=0
col=[]

for a in range(0,col_len,1):
 type_collom=type(data_train_col[0,a])
 if  (type_collom == str):  # de
  col=np.append(col, int(count))
 count=count+1

col = list(map(int, col))
col_re = list(data_train.columns.values)

name= []
for a in col:
 name.append(col_re[a])

for a in name:
 all_names = set(data_train[a]) # replace N/A by A
 unique = {k: i for i, k in enumerate(all_names)} # create unique names for replace str 
 data_train[a] = data_train[a].map(unique)

data_train=data_train.values
x = data_train[:,1:79]
y = data_train[:,80]

x = x.astype(int)
y = y.astype(int)

#2. Data processing
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
x = min_max_scaler.fit_transform(x)

y= y.reshape(-1, 1)
y = min_max_scaler.fit_transform (y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)

#3. Model
#3.1 Create model
model = Sequential()
model.add(Dense(200, input_dim=78, kernel_initializer='normal', activation='relu')) # input_dim=78,
model.add(Dense(100, kernel_initializer='normal', activation='relu'))
model.add(Dense(50, kernel_initializer='normal', activation='relu'))
model.add(Dense(25, kernel_initializer='normal', activation='relu')) #activation='softmax'   activation='sigmoid'
model.add(Dense(1, kernel_initializer='normal')) 
model.summary()

#visualizer(model, format='png', view=True)

#3.2 Compile
model.compile(loss='mean_squared_error', metrics=['acc'], optimizer=keras.optimizers.Adadelta())
#model.compile(optimizer=tf.train.AdamOptimizer(),loss='mse', metrics=['acc'])

#3.3 Train
#early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
hist = model.fit (x_train, y_train,  batch_size = 32 , epochs = 100) #history = m
#model.fit(validation_split=0.1)
#3.4 Evaluate the model

print (model.evaluate (x_test, y_test))

import matplotlib.pyplot as plt
plt.plot(hist.history['loss'])
plt.plot(hist.history['acc'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('acc')
plt.legend(['Loss', 'acc'], loc='upper right')
plt.show()

model.predict(x_test)
              
              
