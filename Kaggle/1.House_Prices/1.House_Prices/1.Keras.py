import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

# 1 data prepairing
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
 unique = {k: i for i, k in enumerate(all_names)}
 data_train[a] = data_train[a].map(unique)

data_train=data_train.values

x = data_train[:,1:79]
y = data_train[:,80]

x = x.astype(np.int)
y = y.astype(np.int)


from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
x = min_max_scaler.fit_transform(x)
#y = min_max_scaler.fit_transform (y)
print (x)



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)







from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
seed = 7
np.random.seed(seed)
# Model
from keras.models import Sequential
model = Sequential()
model.add(Dense(200, input_dim=78, kernel_initializer='normal', activation='relu'))
model.add(Dense(100, kernel_initializer='normal', activation='relu'))
model.add(Dense(50, kernel_initializer='normal', activation='relu'))
model.add(Dense(25, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
# Compile model

model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adadelta())
hist = model.fit (x_train, y_train,  batch_size = 32 , epochs = 100)

# Evaluation on the test set created by train_test_split
print (model.evaluate (x_test, y_test))

