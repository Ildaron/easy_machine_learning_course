import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv("dataset.csv")
plt.xlabel("nm, data from PIA25", fontsize=8, fontweight="bold")
#plt.ylabel("Ось Y\nЗависимая величина", fontsize=8, fontweight="bold")
z = (dataset[" Elapsed"])
plt.title('Data analysis')
plt.plot(dataset["Date"],dataset[" Elapsed"],dataset["Sigma X(um)"],label="Sigma")# ,'b' # blue markers with default shape 'ro' # red circles 'g-', '--' 

plt.legend()
plt.show()


  
