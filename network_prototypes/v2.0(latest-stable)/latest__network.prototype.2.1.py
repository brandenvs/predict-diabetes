import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# Constants
SHUFFLE_BUFFER = 500
BATCH_SIZE = 32

# Read CSV and create a DataFrame
def create_dframe(csv_path):
    df = pd.read_csv(csv_path, engine="python")
    print('Dataset Dimensions (No.Entries, No.Columns):\n', df.shape)
    return df

def preprocess_data(df):
    # Handle missing values
    df = df.fillna(method='ffill')  # Forward-fill missing values

    # Encode categorical columns
    encoder = LabelEncoder()
    encoded_features = [feat for feat in df.columns if df[feat].dtypes == 'object']
    df[encoded_features] = df[encoded_features].apply(encoder.fit_transform)

    return df

# Load and preprocess data
df = create_dframe('network_prototypes/data/diabetes.csv')
df = preprocess_data(df)

# Define target and features
target = df.pop('weight')
features = df

# Create a TensorFlow dataset
dataset = tf.data.Dataset.from_tensor_slices((features.values, target.values))
dataset = dataset.shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(features.shape[1],)),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
model.fit(dataset, epochs=10)

# Save the model
model.save('network_prototypes/model__1.0alpha(latest-stable)')
