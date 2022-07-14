import pandas as pd
dataset = pd.DataFrame({"beam_1": [],"beam_2": [],"mount_1_1": [],"mount_1_2": [],"mount_2_1": [],"mount_2_2": []})
data=[1,2,3,4,5,6]
dataset.loc[0] = data
dataset.loc[1] = data
print(dataset)
