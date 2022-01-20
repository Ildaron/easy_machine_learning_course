import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.datasets import fashion_mnist
import keras_tuner
from keras_tuner import RandomSearch
import keras

from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from sklearn.metrics import mean_squared_log_error as msle, mean_squared_error as mse, make_scorer

print ("ok")
x_train = pd.read_excel('x_train.xls').values
y_train = pd.read_excel('y_train.xls').values
x_test = pd.read_excel('x_test.xls').values
y_test = pd.read_excel('y_test.xls').values
print ("ok1")

x_train = np.delete(x_train, 0, 1)
y_train = np.delete(y_train, 0, 1)
y_test = np.delete(y_test, 0, 1)
x_test = np.delete(x_test, 0, 1)

print ("x shape", x_test.shape)
model = Sequential()
model.add(Dense(1000, input_dim=78, kernel_initializer='normal', activation='relu')) 
model.add(Dense(750, kernel_initializer='normal', activation='relu'))
model.add(Dense(500, kernel_initializer='normal', activation='relu'))
model.add(Dense(250, kernel_initializer='normal', activation='relu'))
model.add(Dense(100, kernel_initializer='normal', activation='relu'))
model.add(Dense(50, kernel_initializer='normal', activation='relu'))
model.add(Dense(25, kernel_initializer='normal', activation='relu')) #activation='softmax'   activation='sigmoid'
model.add(Dense(1, kernel_initializer='normal')) 
model.summary()
model.compile(optimizer='rmsprop', loss='mse',metrics=['mean_squared_logarithmic_error'])

hist = model.fit (x_train, y_train,  batch_size = 32, epochs = 1)
model.evaluate(x_test,y_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Flatten,Dropout,Dense
from tensorflow.keras.optimizers import RMSprop

def build_model(hp):        
    model=Sequential()
    #model.add(Dense(200, input_dim=78, kernel_initializer='normal', activation='relu'))
    #model.add(Dense(units=hp.Int('num_of_neurons',min_value=5,max_value=500,step=5),activation='relu'))  #!!!!
    for i in range(hp.Int('num_of_layers',2,20)):         
        #providing range for number of neurons in hidden layers
     model.add(Dense(units=hp.Int('num_of_neurons'+ str(i),min_value=5,max_value=500,step=1),activation='relu'))
     
    model.add(Dense(1, kernel_initializer='normal')) 
    model.compile(optimizer='rmsprop', loss='mse',metrics=['mean_squared_logarithmic_error'])    
    return model

tuner=RandomSearch(build_model,
    objective='mean_squared_logarithmic_error',
    max_trials=5,
    executions_per_trial=2,
    directory='tuner1',
    project_name='Clothing')

#hp for 2 = neurons,learning rate
tuner.search_space_summary() 
tuner.search(x_train,y_train,epochs=1,validation_data=(x_test,y_test)) 
tuner.results_summary()
