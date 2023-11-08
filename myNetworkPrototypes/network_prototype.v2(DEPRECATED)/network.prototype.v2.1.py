'''
Hello and Welcome to my Neural Network Model which 
implements backpropagation when training.

CODE IMPLEMENTED BY BRANDEN VAN STADEN
'''
# IMPORTS
import pandas as pd
import os
import numpy as np
from ydata_profiling import ProfileReport
import tensorflow as tf
from keras import layers, Sequential
from sklearn.preprocessing import LabelEncoder

# Constants
SHUFFLE_BUFFER = 500
BATCH_SIZE = 2
MODEL_PATH = 'network_prototypes/v2.0(latest-stable)/model'

# Read CSV and create a DataFrame
def create_dframe(csv_path=str()):
    df = pd.read_csv(csv_path, engine="python")
    print('Dataset Dimensions (No.Entries, No.Columns):\n', df.shape)
    print('Press [ENTER] to continue...')
    input()
    return df

# Function to create a TensorFlow dataset
def create_dataset(normal_dframe=pd.DataFrame(), target_col=str()):
    target = normal_dframe.pop(target_col)
    features = normal_dframe
    dataset = tf.data.Dataset.from_tensor_slices((features.values, target.values))
    dataset = dataset.shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE)
    # trainable_tensor_dataset = tf.convert_to_tensor(dataset)
    return dataset

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
    jupyter_report.to_file(output_file='network_prototypes\\reports\\diabetes_report.html')
    return jupyter_report

# Normalization Layer & Preprocessing Model
def normalize_dframe(dframe=pd.DataFrame()):
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

# Implement a Trainable Neural Network Backpropagation Model
def train_model(normalized_dframe, target):    
    # Create a new model or load the existing model if it already exists
    model = None
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print('Loaded existing model.')
    
    if model is None:        
        normalizer = tf.keras.layers.Normalization(axis=-1)
        normalizer.adapt(normalized_dframe)
        print(normalizer(normalized_dframe.iloc[:5]))

        trainable_tensor_dataset = tf.convert_to_tensor(normalized_dframe)

        # Define  features
        features = normalized_dframe

        # Create a TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices((features.values, target))
        dataset = dataset.shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE)

        # Define the sequential model
        model = Sequential([
            normalizer,
            layers.Dense(10, activation='relu'),
            layers.Dense(10, activation='relu'),
            layers.Dense(1)
        ])

        # Compile the model - 
        model.compile(optimizer='adam', # NOTE IMPORTANT: Backpropagation is implement by calling the optimizer 'adam'
                    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), # Configure Loss functions
                    metrics=['accuracy']) # Provide accuracy metrics to observe models progression

    # Train the model and save model to local model project folder
    model.fit(dataset, epochs=10, batch_size=BATCH_SIZE)

    # Save the model to model project folder
    model.save('network_prototypes/model__2.0(latest-stable)/model')    
    return model

print('\n---------------\nWelcome to my Neural Network Backpropagation Model based on the 10 Years Diabetes Dataset\n---------------\n')
print()
print('Currently Loading Diabetes Dataset into Data Frame, PLEASE WAIT...')
# Load and preprocess data
_dframe = create_dframe('network_prototypes/data/diabetes.csv')
print('\n---------------\nSUCCESSFULLY LOADED CSV FILE INTO PANDAS DATA FRAME!\n---------------\n')
print('Generating Statistics for Diabetes Dataset, PLEASE WAIT...')
print('Press [ENTER] to continue generating...')
input()
# report = overview_data(_dframe)
print('\n---------------\nSUCCESSFULLY GENERATED STATISTICS FOR DATASET!\n---------------\n')
print('(Find an HTML File visualizing the Dataset just loaded. FIND REPORT AT: network_prototypes\\v2.0(latest-stable)\\reports\\diabetes_report.html)')
print('Press [ENTER] to continue...')
input()

# Normalization Layer
normalized_dframe = normalize_dframe(_dframe)
print(normalized_dframe.info())

target_weight = normalized_dframe.pop('weight')

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(normalized_dframe)
print(normalizer(normalized_dframe.iloc[:5]))

# Defining features
features = normalized_dframe
# Dictionary of features based on the Diabetes Data Frame
feature_columns = {
    'encounter_id': _dframe['encounter_id'],
    'patient_nbr': _dframe['patient_nbr'],
    'race': _dframe['race'],
    'gender': _dframe['gender'],
    'age': _dframe['age'],
    'weight': _dframe['weight'],
    'admission_type_id': _dframe['admission_type_id'],
    'discharge_disposition_id': _dframe['discharge_disposition_id'],
    'admission_source_id': _dframe['admission_source_id'],
    'time_in_hospital': _dframe['time_in_hospital'],
    'payer_code': _dframe['payer_code'],
    'medical_specialty': _dframe['medical_specialty'],
    'num_lab_procedures': _dframe['num_lab_procedures'],
    'num_procedures': _dframe['num_procedures'],
    'num_medications': _dframe['num_medications'],
    'number_outpatient': _dframe['number_outpatient'],
    'number_emergency': _dframe['number_emergency'],
    'number_inpatient': _dframe['number_inpatient'],
    'diag_1': _dframe['diag_1'],
    'diag_2': _dframe['diag_2'],
    'diag_3': _dframe['diag_3'],
    'number_diagnoses': _dframe['number_diagnoses'],
    'max_glu_serum': _dframe['max_glu_serum'],
    'A1Cresult': _dframe['A1Cresult'],
    'metformin': _dframe['metformin'],
    'repaglinide': _dframe['repaglinide'],
    'nateglinide': _dframe['nateglinide'],
    'chlorpropamide': _dframe['chlorpropamide'],
    'glimepiride': _dframe['glimepiride'],
    'acetohexamide': _dframe['acetohexamide'],
    'glipizide': _dframe['glipizide'],
    'glyburide': _dframe['glyburide'],
    'tolbutamide': _dframe['tolbutamide'],
    'pioglitazone': _dframe['pioglitazone'],
    'rosiglitazone': _dframe['rosiglitazone'],
    'acarbose': _dframe['acarbose'],
    'miglitol': _dframe['miglitol'],
    'troglitazone': _dframe['troglitazone'],
    'tolazamide': _dframe['tolazamide'],
    'examide': _dframe['examide'],
    'citoglipton': _dframe['citoglipton'],
    'insulin': _dframe['insulin'],
    'glyburide.metformin': _dframe['glyburide.metformin'],
    'glipizide.metformin': _dframe['glipizide.metformin'],
    'glimepiride.pioglitazone': _dframe['glimepiride.pioglitazone'],
    'metformin.rosiglitazone': _dframe['metformin.rosiglitazone'],
    'metformin.pioglitazone': _dframe['metformin.pioglitazone'],
    'change': _dframe['change'],
    'diabetesMed': _dframe['diabetesMed']
}

# DataFrame of features used 
features = pd.DataFrame(feature_columns)

print(f'Training Model with TARGET >>> weight.')
weight_model = train_model(normalized_dframe, target_weight)
print(f'Training Model with TARGET >>> age.')
train_model(normalized_dframe, 'age')
