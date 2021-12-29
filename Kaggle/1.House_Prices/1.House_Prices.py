import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


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
   # replace N/A by A
 all_names = set(data_train[a])
 unique = {k: i for i, k in enumerate(all_names)}
 data_train[a] = data_train[a].map(unique)

#data_train=data_train.values
np.set_printoptions(threshold=np.inf)  



#data_train=data_train.fillna(1)

data_train=data_train.values
x = data_train[:,1:79]
y = data_train[:,80]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)

x_train = x_train.astype(np.int)
y_train = y_train.astype(np.int)
#print (len(x_train))
#print (len(y_train))



# 2 Data processing

#3. Machine learning
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

#from sklearn.tree import RandomForestClassifier

# fit a CART model to the data
model = RandomForestClassifier(n_estimators=10, min_samples_split=3)

                                  # class sklearn.ensemble.RandomForestClassifier(n_estimators=10,
                                  #                                              criterion='gini', 
                                  #                                               max_depth=None,
                                  #                                               min_samples_split=2,
                                  #                                               min_samples_leaf=1, 

                                  #                                               min_weight_fraction_leaf=0.0, 
                                  #                                               max_features='auto', 
                                  #                                               max_leaf_nodes=None, 
                                  #                                               bootstrap=True, 
                                  #                                               oob_score=False,
                                  #                                               n_jobs=1, 
                                  #                                               random_state=None,
                                  #                                               verbose=0, 
                                  #                                               warm_start=False, 
                                  #                                               class_weight=None)


model.fit(x_train, y_train)


#print(model)
# make predictions

expected = y_test
predicted = model.predict(x_test)
# summarize the fit of the model
#print(metrics.classification_report(expected, predicted))
#print(metrics.confusion_matrix(expected, predicted))
from sklearn.metrics import accuracy_score  

print (type(predicted))
print ("ok")
print (type(y_test))

z=accuracy_score(list(y_test), list(predicted))
print (z)
