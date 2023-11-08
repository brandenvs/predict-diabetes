'''
PROTOTYPE VERSION 2.0
--
CODE IMPLEMENTED BY BRANDEN VAN STADEN
'''
# IMPORTS
import pandas as panda
import numpy as np
import tensorflow as tf
from keras import layers
from ydata_profiling import ProfileReport
from sklearn.preprocessing import LabelEncoder

# CONSTANTS
SHUFFLE_BUFFER = 500
BATCH_SIZE = 2

# Read CSV Rows and Create a DataFrame
def create_dframe(csv_path):
  # Read CSV using panda
  panda_dframe = panda.read_csv(csv_path, engine="python")

  # Prints FIRST 4 rows of the DataFrame
  print('\n----------------------------------------------\nFIRST 4 rows of Dataset:\n')
  print(panda_dframe.head())
  print('\n')
  
  # Prints LAST 4 rows of the DataFrame
  print('\n----------------------------------------------\nLAST 4 rows of Dataset:\n')
  print(panda_dframe.tail())
  print('\n')

  # Prints the dimensions of the DataFrame
  print('\n----------------------------------------------\nDataset Dimensions(No.Entries, No.Columns):\n')
  print(panda_dframe.shape)

  # Prints some details about the DataFrame
  print('\n----------------------------------------------\nBelow is a table which shows some details about our dataset. \nIf a Dtype is not uniform we must address it before preprocessing model can be built...\n')
  print(panda_dframe.info())
  print('\n')

  # TAKE INTO SEPARATE FUNC.
  # Generate a report from the panda Data Frame. Output is HTML code rendered within a Jupyter notebook
  #!!! NOTE REMOVE COMMENT !!!# jupyter_report = ProfileReport(panda_dframe, title = "10 Year Diabetes Dataframe - Pandas Profiling Report", dark_mode=True, html={'style':{'full_width':True}})
  # Save HTML output to diabetes_report.html file
  #!!! NOTE REMOVE COMMENT !!!# jupyter_report.to_file(output_file='network_prototype/reports/diabetes_report.html')
  return panda_dframe

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

_dframe = create_dframe('network_prototypes/data/diabetes.csv')
print('SUCCESSFULLY CREATED REPORT!\n>>>Find report network_prototype\\reports\\diabetes_report.html to see the report!')

# Normalization Layer
normalized_dframe = normalize_dframe(_dframe)
print(normalized_dframe.info())

target = normalized_dframe.pop('weight')
inputs_dict = {}
features_dict = {}

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(normalized_dframe)
print(normalizer(normalized_dframe.iloc[:5]))

normalized_dframe = tf.convert_to_tensor(normalized_dframe)

diabetes_model = tf.keras.Sequential([
    normalizer,
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
  ])

diabetes_model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])

diabetes_model.fit(normalized_dframe, target, epochs=10, batch_size=BATCH_SIZE)
diabetes_model.save('network_prototypes/1.5(stable)/model')
print(diabetes_model.get_params().keys())