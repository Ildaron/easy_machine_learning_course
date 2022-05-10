
from sklearn import datasets
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

iris_df = datasets.load_iris()

model = TSNE(learning_rate=100)

transformed = model.fit_transform(iris_df.data)
x_axis = transformed[:, 0]
y_axis = transformed[:, 1]

plt.scatter(x_axis, y_axis, c=iris_df.target)
plt.show()
