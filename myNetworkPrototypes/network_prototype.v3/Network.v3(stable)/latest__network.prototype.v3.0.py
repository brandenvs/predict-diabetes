# IMPORTS
import pandas as panda
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from ydata_profiling import ProfileReport

# CONSTANTS
SHUFFLE_BUFFER = 500
BATCH_SIZE = 32

# Function Usage: Create Data Frame using Pandas
def create_dframe(csv_path=str()):
    dframe = panda.read_csv(csv_path, engine="python")
    print('Dataset Dimensions (No.Entries, No.Columns):\n', dframe.shape)
    return dframe

# Function Usage: Preprocessing of Data Frame
def tensor_slicer(dframe=panda.DataFrame(), target=panda.Series()):
    # NOTE TRAINING MODEL & TENSOR_FLOW DATASET SLICING(optimization)
    # Define Features of Dataset after Label has been dropped 
    features = dframe

    # Create a new dataset which contains Tensor slices
    sliced_dframe = tf.data.Dataset.from_tensor_slices((features.values, target.values))
    sliced_dframe = sliced_dframe.shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE)

    return sliced_dframe

# NOTE Uncertain if this is the correct func. name. 
# Function Usage: This Function makes a copy of Data Frames feature, Define parameters for the Diabetes Model and Compiles the Diabetes Model.
def preprocessing_model(dframe=panda.DataFrame()):
    # Make copy of dframe
    features = dframe.copy()
    
    # Setup diabetes model
    diabetes_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(features.shape[1],)), # Adjusts the model's shape based on Data Frame Features
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    # Compile diabetes model
    diabetes_model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
    return diabetes_model

# Function Usage: Normalize Data Frame
def normalize_dframe(dframe=panda.DataFrame()):
    # Make a copy of the data frame
    normalized_dframe = dframe.copy()

    print('\n-------------------\nBEFORE NORMALIZATION LAYER:\n')
    normalized_dframe.info()
    print('Once you have inspected the data frame before normalization layer has been applied PRESS [ENTER] TO CONTINUE...')
    input()

    # Handle missing values
    normalized_dframe = normalized_dframe.fillna(method='ffill')  # Forward-fill missing values

    normalized_dframe = normalized_dframe.where(~normalized_dframe.race.isnull(),normalized_dframe.fillna(axis=0,method='ffill'))
    # #
    normalized_dframe = normalized_dframe.where(~normalized_dframe.payer_code.isnull(),normalized_dframe.fillna(axis=0,method='ffill')) 
    # #
    normalized_dframe = normalized_dframe.where(~normalized_dframe.medical_specialty.isnull(),normalized_dframe.fillna(axis=0,method='ffill'))
    # #
    normalized_dframe = normalized_dframe.where(~normalized_dframe.diag_2.isnull(),normalized_dframe.fillna(axis=0,method='ffill'))
    # #
    normalized_dframe = normalized_dframe.where(~normalized_dframe.diag_3.isnull(),normalized_dframe.fillna(axis=0,method='ffill'))

    # Handle Missing entires for weight with the mode value
    normalized_dframe['weight'] = normalized_dframe['weight'].fillna(normalized_dframe['weight'].mode()[0]) # 
    # Handle Ranges for age entires
    normalized_dframe['age'] = normalized_dframe.age.str.extract('(\d+)-(\d+)').astype('int').mean(axis=1).astype('int') # 

    # Encoding Data Frame features
    encoder = LabelEncoder()
    encoded_features = [feat for feat in normalized_dframe.columns if normalized_dframe[feat].dtypes == 'object']
    normalized_dframe[encoded_features] = normalized_dframe[encoded_features].apply(encoder.fit_transform)
    
    print('\n-------------------\nAFTER NORMALIZATION LAYER:\n')
    normalized_dframe.info()
    print('Once you have inspected the data frame AFTER normalization layer has been applied PRESS [ENTER] TO CONTINUE...')
    input()

    # Return the normalized DATA FRAME
    return normalized_dframe

# Function Usage: Generate Data Statistics & Render HTML Overview of Data Frame
def overview_data(dframe=panda.DataFrame()):
    # Prints FIRST 4 rows of the DataFrame
    print('\n----------------------------------------------\nFIRST 4 rows of Dataset:\n')
    print(dframe.head())
    print('\n')
    
    # Prints LAST 4 rows of the DataFrame
    print('\n----------------------------------------------\nLAST 4 rows of Dataset:\n')
    print(dframe.tail())
    print('\n')

    # Prints the dimensions of the DataFrame
    print('\n----------------------------------------------\nDataset Dimensions(No.Entries, No.Columns):\n')
    print(dframe.shape)

    # Prints some details about the DataFrame
    print('\n----------------------------------------------\nBelow is a table which shows some details about our dataset. \nIf a Dtype is not uniform we must address it before preprocessing model can be built...\n')
    print(dframe.info())
    print('\n')

    # TAKE INTO SEPARATE FUNC.
    # Generate a report from the panda Data Frame. Output is HTML code rendered within a Jupyter notebook
    jupyter_report = ProfileReport(dframe, title = "10 Year Diabetes Dataframe - Pandas Profiling Report", dark_mode=True, html={'style':{'full_width':True}})
    # Save HTML output to diabetes_report.html file
    jupyter_report.to_file(output_file='myNetworkPrototypes\\reports\\diabetes_report.html')
    return jupyter_report


# Terminal Menu
def menu():
    pass

# NOTE DATA FRAME SETUP & NORMALIZATION LAYER
# Get Data Frame
_dframe = create_dframe('myNetworkPrototypes/datasets/diabetes.csv')
# Normalize Data Frame
_dframe = normalize_dframe(_dframe)

# NOTE DATA FRAME GENERATE REPORT & PREPROCESSING MODEL
# Generate Data Frame Report
# report = overview_data(_dframe)


# TESTING THE MODEL #
# Define Target Series
_target = _dframe.pop('weight')
# Pass Normalized Data Frame & Target Series to Tensor Sliced Objects 
_sliced_dframe = tensor_slicer(_dframe, _target)

# Get the Diabetes Model
diabetes_model = preprocessing_model(_dframe)

# Train diabetes model 
# NOTE  I use the slices for an effective highly efficient approach(consider the size of our dataset. 
#       This could be the only way to local train!)
diabetes_model.fit(_sliced_dframe, epochs=3)

# Save diabetes model to project folder(myNetworkPrototypes\network_prototype.v3\model)
try:
    diabetes_model.save('myNetworkPrototypes\\network_prototype.v3\\model\\(trained)diabetes_model.v3')
    print('MODEL SUCCESSFULLY TRAINED & SAVED TO NETWORKS DIRECTORY\n(myNetworkPrototypes\\network_prototype.v3\\model\\diabetes_model.v3))')
except:
    print('error: MODEL WAS UNABLE TO SAVE')