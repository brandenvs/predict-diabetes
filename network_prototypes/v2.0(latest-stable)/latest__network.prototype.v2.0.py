'''
Hello and Welcome to my Neural Network Model which 
implements backpropagation when training.

CODE IMPLEMENTED BY BRANDEN VAN STADEN
'''
# IMPORTS
import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
import tensorflow as tf
from keras import layers, Sequential
from sklearn.preprocessing import LabelEncoder

# CONSTANTS
SHUFFLE_BUFFER = 500
BATCH_SIZE = 32

# Read CSV and create a DataFrame
def create_dframe(csv_path=str()):
    df = pd.read_csv(csv_path, engine="python")
    print('Dataset Dimensions (No.Entries, No.Columns):\n', df.shape)
    return df

# Generate Data Statistics

def overview_data(df=pd.DataFrame()):
      # Prints FIRST 4 rows of the DataFrame
    print('\n----------------------------------------------\nFIRST 4 rows of Dataset:\n')
    print(df.head())
    print('\n')
    
    # Prints LAST 4 rows of the DataFrame
    print('\n----------------------------------------------\nLAST 4 rows of Dataset:\n')
    print(df.tail())
    print('\n')

    # Prints the dimensions of the DataFrame
    print('\n----------------------------------------------\nDataset Dimensions(No.Entries, No.Columns):\n')
    print(df.shape)

    # Prints some details about the DataFrame
    print('\n----------------------------------------------\nBelow is a table which shows some details about our dataset. \nIf a Dtype is not uniform we must address it before preprocessing model can be built...\n')
    print(df.info())
    print('\n')

    # TAKE INTO SEPARATE FUNC.
    # Generate a report from the panda Data Frame. Output is HTML code rendered within a Jupyter notebook
    jupyter_report = ProfileReport(df, title = "10 Year Diabetes Dataframe - Pandas Profiling Report", dark_mode=True, html={'style':{'full_width':True}})
    # Save HTML output to diabetes_report.html file
    jupyter_report.to_file(output_file='network_prototype/reports/diabetes_report.html')
    return jupyter_report

# Preprocessing Data Frame, Required to Train the Model
def preprocess_data(df=pd.DataFrame()):
    """"""
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
model = Sequential([
    layers.Input(shape=(features.shape[1],)),
    layers.Dense(10, activation='relu'),
    layers.Dense(10, activation='relu'),
    layers.Dense(1)
])

# Compile the model - Backpropagation is implement by calling the optimizer 'adam'
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
model.fit(dataset, epochs=10)

# Save the model
model.save('network_prototypes/2.0(latest-stable)/model')
