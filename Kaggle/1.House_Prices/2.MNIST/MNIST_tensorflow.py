import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data')

first_image = mnist.train.images[1]
first_image = np.array(first_image, dtype='float')
pixels = first_image.reshape((28, 28))
plt.imshow(pixels, cmap='gray')
plt.show()

feature_columns = [tf.feature_column.numeric_column("x", shape=[28, 28])]

# Build 2 layer DNN classifier
# Estimator - 
#tf.estimator.DNNClassifier 
#tf.estimator.DNNLinearCombinedClassifier 
#tf.estimator.LinearClassifier 

classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[256, 32],
    optimizer=tf.train.AdamOptimizer(1e-4),
    n_classes=10,
    dropout=0.1,
    model_dir="./tmp/mnist_model"
)


# Returns input function that would feed dict of numpy arrays into the model.
x={"x": mnist.train.images}

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": mnist.train.images},                     # mnist.train.images
    y= mnist.train.labels.astype(np.int32),          # mnist.train.labels.astype(np.int32)
    num_epochs=None,
    batch_size=50,
    shuffle=True
)


classifier.train(train_input_fn, steps=100)
# Define the test inputs
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": mnist.test.images},
    y=mnist.test.labels.astype(np.int32),
    num_epochs=1,
    shuffle=False
)
y=mnist.test.labels.astype(np.int32)


# Evaluate accuracy
accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
print("\nTest Accuracy: {0:f}%\n".format(accuracy_score*100))

