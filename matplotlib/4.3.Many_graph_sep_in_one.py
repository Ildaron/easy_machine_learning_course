import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("dataset.csv")
col = list(dataset.columns.values)

idx=1 
fig = plt.figure(figsize=(10, 10))
for a in col:
  if a !="Unnamed: 20" and a !="Date":   
    ax = fig.add_subplot(5, 5, idx)    
    idx +=1
    ax.plot(dataset["Date"], dataset[a])
    ax.set_title(a)
fig.tight_layout()
fig.show()
