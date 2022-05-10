import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("dataset.csv")
col = list(dataset.columns.values)

figure, (columns) = plt.subplots(len(col), 1, sharex=True)
axis_ch=columns
b=0
for a in col:
 if a !="Unnamed: 20":   
  print ("a",a)
  axis_ch[b].plot(dataset["Date"],dataset[a])
  b=b+1
  
plt.xlabel("nm, data", fontsize=8, fontweight="bold")
plt.show()


  
