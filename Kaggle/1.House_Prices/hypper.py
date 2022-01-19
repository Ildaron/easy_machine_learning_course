#importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.datasets import fashion_mnist

(X_train,y_train),(X_test,y_test)=fashion_mnist.load_data()

#visualizing the dataset
for i in range(25):
    # define subplot
    plt.subplot(5, 5, i+1)
    # plot raw pixel data
    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
# show the figure
plt.show()

#normalizing the images
X_train=X_train/255
X_test=X_test/255

model=Sequential([
    #flattening the images
    Flatten(input_shape=(28,28)),
    #adding first hidden layer
    Dense(256,activation='relu'),
    #adding second hidden layer
    Dense(128,activation='relu'),
    #adding third hidden layer
    Dense(64,activation='relu'),
    #adding output layer
    Dense(10,activation='softmax')
])

#compiling the model
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#fitting the model
model.fit(X_train,y_train,epochs=10)

#evaluating the model
model.evaluate(X_test,y_test)








def build_model(hp):          #hp means hyper parameters
    model=Sequential()
    model.add(Flatten(input_shape=(28,28)))
    #providing range for number of neurons in a hidden layer
    model.add(Dense(units=hp.Int('num_of_neurons',min_value=32,max_value=512,step=32),
                                    activation='relu'))
    #output layer
    model.add(Dense(10,activation='softmax'))
    #compiling the model
    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate',values=[1e-2, 1e-3, 1e-4])),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return model




#feeding the model and parameters to Random Search
tuner=RandomSearch(build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=3,
    directory='tuner1',
    project_name='Clothing')



#this tells us how many hyperparameter we are tuning
#in our case it's 2 = neurons,learning rate
tuner.search_space_summary()

#fitting the tuner on train dataset
tuner.search(X_train,y_train,epochs=10,validation_data=(X_test,y_test))

tuner.results_summary()

