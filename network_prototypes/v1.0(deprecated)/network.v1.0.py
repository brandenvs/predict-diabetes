'''NOTE NOT WORKING!!!
Hello and Welcome to my Neural Network Model which 
implements backpropagation when training.

CODE IMPLEMENTED BY BRANDEN VAN STADEN
'''
# IMPORTS
import pandas as panda
import numpy as np
import tensorflow as tf
from keras import layers
from ydata_profiling import ProfileReport
from sklearn.preprocessing import LabelEncoder

# Dictionary used to store symbolic input objects
inputs = {}

# Read CSV Rows and Create a DataFrame
def create_dframe(csv_path):
  # Read CSV using panda
  dframe = panda.read_csv(csv_path, engine="python")

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
  print('\n----------------------------------------------\nBelow is a table which shows some details about our dataset. \nThe Dtype is important as this must be preprocessed...\n')
  print(dframe.info())
  print('\n')

  # Profile DataFrame and Print Result Generated
  df_profile = ProfileReport(dframe, title = "10 Year Diabetes Dataframe - Pandas Profiling Report", dark_mode=True)
  print(df_profile)
  
  return dframe

# Preform Feature Engineering to prevent AttributeError Exception
def dframe_bespoke_engineering(dframe=panda.DataFrame()):
  diabetes_dframe = dframe.copy()
  # Replace ?s with constant float
  diabetes_dframe = diabetes_dframe.replace("?", np.NaN)

  # Prints all null/empty fields in dataset(inspection)
  print(diabetes_dframe.isnull().sum())

  # BESPOKE TO 10 YEAR DIABETES DATASET
  # [START]
  diabetes_dframe = diabetes_dframe.where(~diabetes_dframe.race.isnull(),diabetes_dframe.fillna(axis=0,method='ffill')) # 
  diabetes_dframe = diabetes_dframe.where(~diabetes_dframe.payer_code.isnull(),diabetes_dframe.fillna(axis=0,method='ffill')) # 
  diabetes_dframe = diabetes_dframe.where(~diabetes_dframe.medical_specialty.isnull(),diabetes_dframe.fillna(axis=0,method='ffill')) # 
  diabetes_dframe = diabetes_dframe.where(~diabetes_dframe.diag_2.isnull(),diabetes_dframe.fillna(axis=0,method='ffill')) # 
  diabetes_dframe = diabetes_dframe.where(~diabetes_dframe.diag_3.isnull(),diabetes_dframe.fillna(axis=0,method='ffill')) # 
  diabetes_dframe['weight'] = diabetes_dframe['weight'].fillna(diabetes_dframe['weight'].mode()[0]) # 
  diabetes_dframe['age'] = diabetes_dframe.age.str.extract('(\d+)-(\d+)').astype('int').mean(axis=1).astype('int') # 
  # [END]

  # Inspect the newly ADJUSTED DataFrame fields(first 4 rows)
  print('\n----------------------------------------------\nDataFrame Head\nAfter addressing some feature field issues, useful when training ;) \n')
  print(diabetes_dframe.head())

  return diabetes_dframe

# Preform Encoding on the DataFrame
def encoding_engineering(dframe=panda.DataFrame()):
  cf = [col for col in dframe.columns if dframe[col].dtypes=='object']
  encoder = LabelEncoder()
  dframe[cf] = dframe[cf].apply(encoder.fit_transform)
  
  # Inspect the newly encoded dataset
  print('\n----------------------------------------------\nDataFrame Total NULL Entries(after encoding) \n')
  print(dframe.isnull().sum())
  print('\n----------------------------------------------\nDataFrame Head \n')
  print(dframe.head())

  return dframe

# Group and Derive some Insights on DataFrame
def dframe_exploring(dframe=panda.DataFrame()):
  # Group and sort 'diabetesMed' column data entries. This determines if the SIZE of the TRUE/FALSE rows. 
  dframe_findings = dframe.groupby(["diabetesMed"]).size().sort_values(ascending=False) # output displays the count of patients that took their medication and the cou t of patients not taking medication.
  # Print these findings & Return them
  print(dframe_findings)

  return dframe_findings

# Build Preprocessing model
def preprocessing_model(dframe=panda.DataFrame()):
  # Make a copy of DataFrame
  dframe_features = dframe.copy()

  # Iterate over CSV headers and columns
  for header, column in dframe_features.items():
    # Determine Data Type of dframe
    data_type = column.dtype
    if data_type == object:
      data_type = tf.string
    else:
      data_type = tf.float32
    
    # I use the Keras functional API to implement the preprocessed model 
    inputs[header] = tf.keras.Input(shape=(1,), name=header, dtype=data_type)
  
  # Print inputs dictionary for inspection
  for header, tensor_object in inputs.items():
    print(f'HEADER: {header}\nTENSOR OBJECT(mem): {tensor_object}')

  # PREPROCESSING - Concatenate(combined) numeric(float) inputs(tensor objects)
  num_inputs = {_header:_input for _header,_input in inputs.items() if _input.dtype == tf.float32}

  # PREPROCESSING - Apply Normalization Layer
  x = layers.Concatenate()(list(num_inputs.values()))
  make_me_normal = layers.Normalization()
  make_me_normal.adapt(np.array(dframe[num_inputs.keys()])) # NOTE I USE THE ORIGINAL DATAFRAME RATHER THAN THE COPIED ONE.
  all_num_inputs = make_me_normal(x)

  # PREPROCESSING - Collect symbolic results and Begin preprocessing of string inputs
  preprocessed_inputs = [all_num_inputs]
  # Preprocess strings
  for header, column in inputs.items():
    if column.dtype == tf.float32:
      continue

    lookup = layers.StringLookup(vocabulary=np.unique(dframe_features[header]))
    one_hot = layers.CategoryEncoding(num_tokens=lookup.vocabulary_size())

    x = lookup(column)
    x = one_hot(x)
    preprocessed_inputs.append(x)

  # FINAL PREPROCESSING - Concatenate all preprocessed inputs and Begin to build model(which will handle the preprocessing itself I believe...)
  preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)
  # PREPROCESSED - TensorFlow Model
  dframe_preprocessing_model = tf.keras.Model(inputs, preprocessed_inputs_cat)

  # AND Return...The Preprocessing Model!
  return dframe_preprocessing_model


print('\nHELLO WORLD!\n--- NEURAL NETWORK VERSION 1 ---\n--- created by: Branden van Staden ---\n')
_dframe1 = create_dframe('v1.0\data\diabetes.csv')
print('\n-------------------------------\nLOADED CSV FILE SUCCESSFULLY!\n-----')
print('Press enter to begin TASK: BESPOKE ENGINEERING...') 
input()
_dframe = dframe_bespoke_engineering(_dframe1)
print('\n-------------------------------\nAPPLIED CHANGES TO DATAFRAME SUCCESSFULLY!\n-----')
print('Press enter to begin TASK: ENCODING...')
_dframe = encoding_engineering(_dframe)
print('\n-------------------------------\nAPPLIED CHANGES TO DATAFRAME SUCCESSFULLY!\n-----')
print('Press any key to begin BUILD TASK: PREPROCESSING MODEL...')
input()
dframe_preprocessing_head = preprocessing_model(_dframe1)
print('\n-------------------------------\nSUCCESS! READY TO BEGIN TRAINING USING THE PREPROCESSING MODEL INPUTS....\n-----')
print('Press any key to begin TRAINING THE MODEL...')

# Make a copy of DataFrame
_dframe_features = _dframe.copy()

# Pop label from DataFrame
dframe_labels = _dframe_features.pop('diabetesMed')


def diabetic_model(preprocessing_head, inputs):
  body = tf.keras.Sequential([
    layers.Dense(64),
    layers.Dense(1)
  ])

  preprocessed_inputs = preprocessing_head(inputs)
  result = body(preprocessed_inputs)
  model = tf.keras.Model(inputs, result)

  model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam())
  return model

diabetes_model_item = diabetic_model(dframe_preprocessing_head, inputs)
features_dict = {_key:_values[:6] for _key, _values in _dframe_features.items()}

diabetes_model_item.fit(x=features_dict, y=dframe_labels, epochs=10)
# Save Model
diabetes_model_item.save('data')
print(diabetes_model_item.get_params().keys())
# Fetch and pop label from DataFrame - Split Test Training Approach
# X_train, X_test, y_train, y_test = train_test_split(_dframe_features, dframe_labels, test_size=0.2, random_state=42)

# Iterate over DataFrame inputs --> Populates KeyValue Pairs(dictionary)
# dframe_features_dict = {key:np.array(value) for key,value in _dframe_features.items()}

# # Slice first training example from dictionary
# 
# # X_train,X_test,y_train,y_test = train_test_split(dframe_labels,dframe_features_dict,test_size=0.2,random_state=42)

# scaler = StandardScaler()

# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)


# Builds a TensorFlow Model - BACK PROPAGATION preprocessing_head inputs
# def train_model(activation='relu'):
#   model = tf.keras.Sequential([layers.Dense(64,input_dim=X_train_scaled.shape[1], activation=activation), layers.Dense(1, activation='sigmoid')])
#   model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


  # body = tf.keras.Sequential([
  #   layers.Dense(64),
  #   layers.Dense(1)
  # ])

  # preprocessed_inputs = preprocessing_head(inputs)
  # result = body(preprocessed_inputs)
  # model = tf.keras.Model(inputs, result)

  # model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
  #               optimizer=tf.keras.optimizers.Adam())
 # return model

# Begin pumping data through training algorithm
# diabetes_model = train_model(dframe_preprocessing_head, inputs)

# Wrapping the model using KerasClassifier
# diabetes_model = KerasClassifier(build_fn=train_model(), verbose=0)


# param_grid = {'units':[34,64,128], 'activation':['relu','tanh']}

# #Creating a RandomizedSearchCV object
# random_search = RandomizedSearchCV(estimator=diabetes_model, param_distributions=param_grid, cv=3, n_iter=5, scoring='accuracy')

# #Performing the hyperparameter search
# random_search.fit(X_train_scaled,y_train)

# #Evaluating the best model using cross-validation
# cross_val_scores = cross_val_score(random_search.best_estimator_,X_test_scaled,y_test,cv=3,scoring='accuracy')

# # Calculating the Logarithmic Loss using log_loss function 
# y_pred_proba = random_search.best_estimator_.predict_proba(X_test_scaled)
# log_loss_scores = log_loss(y_test,y_pred_proba)

# print("Cross-Validation Test Accuracy:",np.mean(cross_val_scores))
# print("Logarithmic Loss:",log_loss_scores)


# Trains the model for a fixed number of epochs || ITERATION OF ENTIRE DATASET!
# diabetes_model.fit(x=features_dict, y=dframe_labels, epochs=2)

# Demonstrates Reloading Functionality
### reloaded = tf.keras.models.load_model('data')
