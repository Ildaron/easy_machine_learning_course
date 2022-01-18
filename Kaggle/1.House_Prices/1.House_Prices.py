# hyperparameter batch_size = {{choice ([32, 64, 128])}} Grid search и Random search.
#Dropout
# GRAPT - ACTUAL AND PREDICTED
# try  Gradient Boosting Regressor # Random Forest Regressor # Ridge regression (L2) # Lasso regression (L1)
# feature selection methods
# Feature Engineering

import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
#from keras_visualizer import visualizer
from sklearn.metrics import mean_squared_log_error as msle, mean_squared_error as mse, make_scorer
import matplotlib.pyplot as plt
print ("library is ok")

#1. Data prepairing
data_train = pd.read_csv('train.csv')
samples_to_predict = pd.read_csv('test.csv')
l=list(range(0,1459)) 
samples_to_predict["C"]= l

def data_all (data_train):
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
 dataset = dict()
 dataset["x"]=x
 dataset["y"]=y
 return dataset

data=data_all(data_train)
x=data["x"]
y=data["y"]

data_test=data_all(samples_to_predict)
data_test_x=data_test["x"]

#2. Data processing 
#2.1 MinMaxScaler
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()


#data_test_x = min_max_scaler.fit_transform(data_test_x)
#x = min_max_scaler.fit_transform(x)
#y= y.reshape(-1, 1)
#y = min_max_scaler.fit_transform (y)


#2.2 Normalization

#normalized_x = preprocessing.normalize(x) #Нормализация предполагает замену номинальных признаков так, чтобы каждый из них лежал в диапазоне от 0 до 1
#standardized_x = preprocessing.scale(x)   # Стандартизация же подразумевает такую предобработку данных, после которой каждый признак имеет среднее 0 и дисперсию 1.

#2.3 Standartization
                    #standardization can be useful and even necessary in some machine learning algorithms when your data has input values at different scales.
from numpy import asarray
from sklearn.preprocessing import StandardScaler
standardScaler_scaler = StandardScaler()

scaler_x = StandardScaler().fit(x)
y= y.reshape(-1, 1)
scaler_y = StandardScaler().fit(y)
data_test_x_scaler = StandardScaler().fit(data_test_x )

x = scaler_x.fit_transform(x)
y = scaler_y.fit_transform(y)
data_test_x = data_test_x_scaler.fit_transform(data_test_x)

#2.4 Split data and Recursive Feature Elimination
x_train, x_test, y_train, y_test = train_test_split(x, y.ravel(), test_size=0.01, random_state=42)

from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.feature_selection import SelectKBest,f_regression,RFECV

mdl7 = GradientBoostingRegressor(n_estimators = 150)
rfecv = RFECV(estimator=mdl7, step=1, cv=5,scoring='neg_mean_squared_error')  
rfecv = rfecv.fit(x, y.ravel())

print('Optimal number of features :', rfecv.n_features_)
print('Best features :',rfecv.support_)    
col_false=rfecv.support_
b=0
c=[]

for a in col_false:
 b=b+1 
 if (str(a)=="False"):        
  c.append(b-1)

x_train = np.delete(x_train, c, 1)
x_test =  np.delete(x_test, c, 1)
data_test_x = np.delete(data_test_x, c, 1)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score of number of selected features")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
#plt.show()


from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier


#3. Model
#3.1 Create model
model = Sequential()
model.add(Dense(200, input_dim=x_train.shape[1], kernel_initializer='normal', activation='relu')) 
model.add(Dense(100, kernel_initializer='normal', activation='relu'))
model.add(Dense(50, kernel_initializer='normal', activation='relu'))
model.add(Dense(25, kernel_initializer='normal', activation='relu')) #activation='softmax'   activation='sigmoid'
model.add(Dense(1, kernel_initializer='normal')) 
model.summary()
#visualizer(model, format='png', view=True)

#3.2 Compile
model.compile(optimizer='rmsprop', loss='mse',metrics=['mean_squared_logarithmic_error','mean_absolute_error', 'mean_absolute_percentage_error','mean_squared_error','mean_absolute_percentage_error', 'cosine_proximity']) # metrics=['acc'],   metrics=['mean_absolute_error']  #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])   - Метрика accuracy предназначена для задачи классификации,
                                                      #   'cosine_proximity'
    #optimizer
    #rmsprop
    #SGD
    #RMSprop
    #Adam
    #Adadelta
    #Adagrad
    #Adamax
    #Nadam
    #Ftrl

#3.3 Train
#early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)

hist = model.fit (x_train, y_train,  batch_size = 32, epochs = 1000)
#model.fit(X_train, y_train, batch_size = batch_size, nb_epoch = nb_epochs, show_accuracy = True, verbose = 2, validation_data = (X_test, y_test), class_weight=classWeight)


#model.fit(validation_split=0.1)

#3.4 Evaluate the model
#test_mse_score, test_mae_score = (model.evaluate (x_test, y_test))

predictions_test = model.predict(data_test_x)
predictions = model.predict(x_test)

#3.5 Inverse y data

#predictions_test = min_max_scaler.inverse_transform(predictions_test)

predictions_test = scaler_y.inverse_transform(predictions_test)
dframe = pd.DataFrame(predictions_test) 
dframe.to_excel('./teams.xlsx')

#print("loni", tf.keras.metrics.mean_squared_logarithmic_error(y_test, predictions))
#print ("msle", msle(predictions, y_test))

#4.Visualization 

figure, axis = plt.subplots(2, 1)
plt.subplots_adjust(hspace=1)

axis[0].plot(hist.history['loss'])
axis[0].plot(hist.history['mean_squared_logarithmic_error'])  #metrics=['mean_absolute_error']
axis[0].plot(hist.history['mean_absolute_error'])
axis[0].plot(hist.history['mean_squared_error'])
#axis[0].plot(hist.history['cosine_proximity'])

axis[1].plot(hist.history['mean_absolute_percentage_error'])
axis[1].plot(hist.history['mean_absolute_percentage_error'])

axis[0].set_xlabel('epoch')
axis[0].set_ylabel('Error')
axis[0].legend(['mean_squared_logarithmic_error','mean_absolute_error','mean_absolute_percentage_error','mean_squared_error', 'cosine_proximity'], loc='upper right')

#plt.title('Model loss')

axis[1].set_xlabel('epoch')
axis[1].set_ylabel('Error')
axis[1].legend(['mean_absolute_error','mean_absolute_percentage_error'], loc='upper right')

plt.show()
