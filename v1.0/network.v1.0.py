# IMPORTS
import pandas as pd
import numpy as np
import tensorflow as tf
from keras import layers

# Print the installed TensorFlow version
print("TensorFlow version:", tf.__version__) # (desired: TensorFlow version: 2.14.0)
# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

diabetes_train = pd.read_csv('v1.0\data\diabetes.csv')

h = diabetes_train.head()
print(h)

diabetes_features = diabetes_train.copy()
diabetes_labels = diabetes_features.pop('age')
diabetes_features = np.array(diabetes_features)
print(diabetes_features)

diabetes_model = tf.keras.Sequential([
  layers.Dense(64),
  layers.Dense(1)
])

diabetes_model.compile(loss = tf.keras.losses.MeanSquaredError(),
                      optimizer = tf.keras.optimizers.Adam())

diabetes_model.fit(diabetes_features, diabetes_labels, epochs=2)

# Normalization of layers
# NOTE The tf.keras.layers.Normalization layer precomputes the mean 
# and variance of each column, and uses these to normalize the data.
iam_normal = layers.Normalization()
iam_normal.adapt(diabetes_features)
iam_normal_diabetes_layer = tf.keras.Sequential({
    iam_normal,
    layers.Dense(64),
    layers.Dense(1)
})

iam_normal.compile(loss = tf.keras.losses.MeanSquaredError(),
                   iam_optimizer = tf.keras.optimizers.Adam())

iam_normal_diabetes_layer.fit(diabetes_features, diabetes_labels, epochs=2)


# # IMPORTS
# import tensorflow as tf
# 
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# # % matplotlib inline
# from sklearn.model_selection import train_test_split, RandomizedSearchCV
# from sklearn.preprocessing import StandardScaler
# from keras.models import Sequential 
# from keras.layers import Dense
# from scikeras.wrappers import KerasClassifier

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('v1.0\data'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# df = pd.read_csv('v1.0\data\diabetes.csv')

# (x_train, y_train), (x_test, y_test) = df.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0

# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(28, 28)),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(10)
# ])

# predictions = model(x_train[:1]).numpy()

# print(predictions)


