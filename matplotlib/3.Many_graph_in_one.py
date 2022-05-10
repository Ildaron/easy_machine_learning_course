import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv("dataset.csv")
col = list(dataset.columns.values)
b=0
for a in col:
 if a !="Unnamed: 20":   
  print ("a",a)
  plt.plot(dataset["Date"],dataset[a])
  b=b+1
plt.xlabel("nm, data from PIA25", fontsize=8, fontweight="bold")
plt.show()


  
