import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv("dataset.csv")
col = list(dataset.columns.values)

b=0
for a in col:
 if a !="Unnamed: 20":   
  print ("a",a)
  plt.plot(dataset["Date"],dataset[a], label=a)
  b=b+1
plt.legend()  
plt.xlabel("nm, data", fontsize=8, fontweight="bold")
plt.ylabel("see legend", fontsize=8, fontweight="bold")
plt.title('Beamage data analysis')
plt.show()


  
