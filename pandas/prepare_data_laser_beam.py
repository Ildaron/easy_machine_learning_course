import pandas as pd

dataset_after = pd.DataFrame()
datasetos=pd.DataFrame()
columns = {"Elapsed time(sec)":0, "dSigma X(um)":0,
           "dSigma Y (um)":0, "Effective":0, "Diameter(um)":0, "Ellipticity(%)":0,
           "Orientation(degrees)":0, "Centroid X(um)":0, "Centroid y(um)":0, "Peak X(um)":0, "Peak Y(um)":0,  "Peak Saturation level(%)":0,
           "Peak to Average Ratio X":0,
           "Peak to Average Ratio Y":0, "X Baseline Position":0,"Y Baseline Position":0, "Delta Centroid X":0, "Delta Centroid Y":0,
            "Delta Peak X":0, "Delta Peak Y":0}
name = ["Elapsed time(sec)", "dSigma X(um)",
           "dSigma Y (um)", "Effective Diameter(um)", "Ellipticity(%)",
           "Orientation(degrees)", "Centroid X(um)", "Centroid y(um)", "Peak X(um)", "Peak Y(um)"," Peak Saturation level(%)","Peak to Average Ratio X","Peak to Average Ratio Y",
        "X Baseline Position","Y Baseline Position","Delta Centroid X","Delta Centroid Y","Delta Peak X","Delta Peak Y"]

for a in range (1,5,1):
 path = "C:/Users/ir2007/Desktop/dataset/adjusting/2_attempt/2_mirror/4."+str(a)+".txt"
 file = open(path, "r") 
 dataset = pd.DataFrame(file)
 for a in dataset.iloc[9]:
  b=a.split()  

 col=0
 for a in b[5:]:
  if col==18:
   pass   
  columns[name[col]]=a   
  col+=1
 dataset_after = dataset_after.append(columns, ignore_index=True)
 
dataset_after.to_csv("C:/Users/ir2007/Desktop/dataset/adjusting/2_attempt/collect/data_2.4.csv")

print (dataset_after)
