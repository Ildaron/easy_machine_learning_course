import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np
import seaborn as sns
plt.rc('xtick', labelsize=4) 
plt.rc('ytick', labelsize=4)

#1. Data show
#1.1 Graph
cycol = cycle('bgrcmk')
dataset = pd.read_csv("dataset1.2.csv")

dataset["Date"]=np.array( [x for x in range(0,100*len(dataset["Date"]),100)])
col = list(dataset.columns.values)
col=col[:-1]
for a in col:
 if a !="Unnamed: 20" and a !="Date":    
  plt.plot(dataset["Date"],dataset[a], label=a)
  
plt.legend()  
plt.xlabel("nm, data from PIA25", fontsize=8, fontweight="bold")
plt.ylabel("see legend", fontsize=8, fontweight="bold")
plt.title('Beamage data analysis')
#plt.show()

idx=1 
fig = plt.figure(figsize=(10, 10))
for a in col:
  if a !="Unnamed: 20" and a !="Date":   
    ax = fig.add_subplot(5, 5, idx)    
    idx +=1
    #plt.xlabel("x", fontsize=1, fontweight="bold")
    #plt.ylabel("y", fontsize=1, fontweight="bold")
    ax.plot(dataset["Date"], dataset[a], c=next(cycol))
    ax.set_title(a, fontsize=8)
fig.tight_layout()
#fig.show()

#1.2 гистограмму
#1.2.1
n_bins = len(dataset["Date"])
idx=1 
fig = plt.figure(figsize=(10, 10))
for a in col:
  if a !="Unnamed: 20" and a !="Date": 
   ax = fig.add_subplot(5, 5, idx)      
   ax.hist(dataset[a], bins=n_bins)
   ax.set_title(a, fontsize=8)
   idx +=1
#fig.show()
   
#1.2.2 Seaborn
idx=1 
fig1 = plt.figure(figsize=(10, 10))
for a in col:
  if a !="Unnamed: 20" and a !="Date" and a !=" Effective Diameter(um)": 
   ax = fig1.add_subplot(5, 5, idx)      
   print ("a",a)
   #sns.violinplot(x = a, data=dataset)
   sns.histplot(dataset[a]) # distplot
   #ax.set_title(a, fontsize=8)
   idx +=1
fig1.show()

#2. Data Analasy
#2.1 Correlations
def correlation(a):
 return (dataset['Date'].corr( dataset[a].apply(np.log)))
for a in col:
 try:
  if a !="Unnamed: 20" and a !="Date":
   pass   
   #print(a, "correlations",correlation(a))
 except TypeError:
  print ("not used",a)




