import pandas as panda
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from ydata_profiling import ProfileReport

# Constants
SHUFFLE_BUFFER = 500
BATCH_SIZE = 32

# Function Usage: Create Data Frame using Pandas
def create_dframe(csv_path=str()):
    dframe = panda.read_csv(csv_path, engine="python")
    print('Dataset Dimensions (No.Entries, No.Columns):\n', dframe.shape)
    return dframe

# Function Usage: Preprocessing of Data Frame
def preprocess_data(dframe=panda.DataFrame()):
    # Handle missing values
    dframe = dframe.fillna(method='ffill')  # Forward-fill missing values

    # Encode categorical columns
    encoder = LabelEncoder()
    encoded_features = [feat for feat in dframe.columns if dframe[feat].dtypes == 'object']
    dframe[encoded_features] = dframe[encoded_features].apply(encoder.fit_transform)

    return dframe

# Function Usage: Normalize Data Frame
def normalize_dframe(dframe=panda.DataFrame()):
    normalized_dframe = dframe.copy()
    # 
    normalized_dframe = normalized_dframe.replace("?", np.NaN)    

    # 
    print(normalized_dframe.isnull().sum())

    # BESPOKE TO 10 YEAR DIABETES DATASET - BASED ON RELATIONS WITH NON-NULL ENTRIES
    # [START]
    normalized_dframe = normalized_dframe.where(~normalized_dframe.race.isnull(),normalized_dframe.fillna(axis=0,method='ffill')) # 
    #
    normalized_dframe = normalized_dframe.where(~normalized_dframe.payer_code.isnull(),normalized_dframe.fillna(axis=0,method='ffill')) # 
    #
    normalized_dframe = normalized_dframe.where(~normalized_dframe.medical_specialty.isnull(),normalized_dframe.fillna(axis=0,method='ffill')) # 
    #
    normalized_dframe = normalized_dframe.where(~normalized_dframe.diag_2.isnull(),normalized_dframe.fillna(axis=0,method='ffill')) # 
    #
    normalized_dframe = normalized_dframe.where(~normalized_dframe.diag_3.isnull(),normalized_dframe.fillna(axis=0,method='ffill')) # 
    #
    normalized_dframe['weight'] = normalized_dframe['weight'].fillna(normalized_dframe['weight'].mode()[0]) # 
    #
    normalized_dframe['age'] = normalized_dframe.age.str.extract('(\d+)-(\d+)').astype('int').mean(axis=1).astype('int') # 
    # [END]

    # Inspect the newly ADJUSTED DataFrame fields(first 4 rows)
    # print('\n----------------------------------------------\nDataFrame Head\n(normalized_dframe initial and BEFORE passing to normalization layer)\n')
    # print(normalized_dframe.head())

    #  normalized_dframe = normalized_dframe.replace(None, np.NaN)

    encoder = LabelEncoder()
    encoded_features = [feat for feat in normalized_dframe.columns if normalized_dframe[feat].dtypes == 'object']

    normalized_dframe[encoded_features] = normalized_dframe[encoded_features].apply(encoder.fit_transform)
    
    # 
    print(f'\n----------------------------------------------\nTotal Count of NULL values in Data Frame.\n>>>Before Passing to Normalization Layer')
    print(normalized_dframe.isnull().sum())
    print(normalized_dframe.head())    
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

# NOTE DATA FRAME SETUP & NORMALIZATION LAYER
# Get Data Frame
_dframe = create_dframe('myNetworkPrototypes/datasets/diabetes.csv')
# Normalize Data Frame
normal_dframe = normalize_dframe(_dframe)

# NOTE DATA FRAME GENERATE REPORT & PREPROCESSING MODEL
# Generate Data Frame Report
######################### report = overview_data(_dframe) REMOVE hashtags! REMOVE hashtags! REMOVE hashtags! 
# Preprocess Data Frame
_dframe = preprocess_data(_dframe)

# NOTE TRAINING MODEL & TENSORFLOW DATASET SLICING(optimization)
# Define Label which model will train on
target = _dframe.pop('weight')
# Define Features of Dataset after Label has been dropped 
features = _dframe

# Create a new dataset which contains Tensor slices
sliced_dataset = tf.data.Dataset.from_tensor_slices((features.values, target.values))
sliced_dataset = sliced_dataset.shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE)

# Setup diabetes model
diabetes_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(features.shape[1],)), # Adjusts the model's shape
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile diabetes model
diabetes_model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train diabetes model 
# NOTE  I use the slices for an effective highly efficient approach(consider the size of our dataset. 
#       This could be the only way to local train!)
diabetes_model.fit(sliced_dataset, epochs=10)

# Save diabetes model to project folder(myNetworkPrototypes\network_prototype.v3\model)
try:
    diabetes_model.save('myNetworkPrototypes\\network_prototype.v3\\model\\(trained)diabetes_model.v3')
    print('MODEL SUCCESSFULLY TRAINED & IS SAVED TO NETWORKS DIRECTORY(myNetworkPrototypes\\network_prototype.v3\\model\\diabetes_model.v3))')
except:
    print('error: MODEL WAS UNABLE TO SAVE')