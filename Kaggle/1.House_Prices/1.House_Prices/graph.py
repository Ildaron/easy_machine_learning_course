https://seaborn.pydata.org/generated/seaborn.pairplot.html

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import hvplot.pandas
print ("ok")

data_train = pd.read_excel('x_test_after_compil.xlsx')
sns.pairplot(data_train)
