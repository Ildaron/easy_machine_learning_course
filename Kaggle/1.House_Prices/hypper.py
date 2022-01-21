
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.datasets import fashion_mnist
from keras_tuner import RandomSearch
import keras_tuner

x_train = pd.read_excel('x_train_after_compil.xlsx').values
y_train = pd.read_excel('y_train_after_compil.xlsx').values
x_test = pd.read_excel('x_test_after_compil.xlsx').values
y_test = pd.read_excel('y_test_after_compil.xlsx').values

x_train = np.delete(x_train, 0, 1)
y_train = np.delete(y_train, 0, 1)
y_test = np.delete(y_test, 0, 1)
x_test = np.delete(x_test, 0, 1)

print (x_test.shape)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Flatten,Dropout,Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers

def build_model(hp):        
 model = Sequential()
    #model.add(layers.Flatten())
 for i in range(hp.Int("num_layers", 1, 3)):
  model.add( layers.Dense(units=hp.Int(f"units_{i}", min_value=32, max_value=96, step=32), activation=hp.Choice("activation", ["relu", "tanh"]),)) #512
  if hp.Boolean("dropout"):
   model.add(layers.Dropout(rate=0.25))
 model.add(Dense(1, kernel_initializer='normal')) 
 #learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
 model.compile(optimizer='rmsprop', loss='mse',metrics=['mean_squared_logarithmic_error'])
 return model

build_model(keras_tuner.HyperParameters())
tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective="mean_squared_logarithmic_error",
    max_trials=3,
    executions_per_trial=2,
    overwrite=True
)

#hp for 2 = neurons,learning rate
tuner.search_space_summary() 
tuner.search(x_train,y_train,epochs=1,validation_data=(x_test,y_test)) 
tuner.results_summary()

models = tuner.get_best_models(num_models=2)
best_model = models[0]
print (best_model)

# Build the model.
# Needed for `Sequential` without specified `input_shape`.
print ("x_test", x_train)
print ("rons", x_train[0].shape)
best_model.build(input_shape=(0,78,0))  #0, 78, 1  # None, 78  # None, (78,)))
best_model.summary()
tuner.results_summary()






