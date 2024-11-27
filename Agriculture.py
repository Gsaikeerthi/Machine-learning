#!/usr/bin/env python
# coding: utf-8

# In[1]:

import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier



# In[2]:


# Function to categorize features into numeric, categorical, binary and features to drop
def categorize_features(df):
    numeric_features = []
    categorical_features = []
    binary_features = []
    features_to_drop = ['UID', 'RawLocationId', 'TownId', 'DistrictId', 'FarmingCommunityId',
                       'AgriculturalPostalZone', 'ValuationYear', 'TaxOverdueYear',
                       'AgricultureZoningCode', 'OtherZoningCode']
    #Preddefined categorical features that need special encoding
    encoded_categorical_features = [
        'TypeOfIrrigationSystem', 'CropFieldConfiguration', 'FarmClassification',
        'HarvestProcessingType', 'LandUsageType', 'FieldZoneLevel', 'FieldConstructionType'
    ]

    #Classifying features based on datatypes and unique values
    for column in df.columns:
        # Skip features that we know we wanted to drop
        if column in features_to_drop:
            continue
            
        # To get number of unique values and data type
        n_unique = df[column].nunique()
        dtype = df[column].dtype

        if column in encoded_categorical_features:
            categorical_features.append(column)
            continue
    
        # Numerical features
        if dtype in ['int64', 'float64']:
            if n_unique <= 2:  # If unique values are 0,1 or 2 then represent them as binary
                binary_features.append(column)
            else:
                numeric_features.append(column)
                
        # Categorical features
        elif dtype == 'object' or dtype == 'category':
            categorical_features.append(column)

    return numeric_features, categorical_features, binary_features, features_to_drop


# In[3]:


#Function to preprocess training and test datasets
def preprocess_data(df, test_df):
    
    # Calculating the missing percentages
    missing_percentage = df.isnull().mean() * 100
    
    # Identifying columns with more than 90% missing values
    columns_to_drop = missing_percentage[missing_percentage > 90].index
    #print("Columns dropped due to more than 90% missing values:")
    #print(columns_to_drop)
    
    # Dropping the colums that satisfy above condition
    df = df.drop(columns=columns_to_drop, axis=1)
    test_df = test_df.drop(columns=columns_to_drop, axis=1)
    
    # Recalculating the missing percentages for the remaining columns
    remaining_missing_percentage = df.isnull().mean() * 100
    column_info = pd.DataFrame({
        'Missing_Percentage': remaining_missing_percentage,
        'Data_Type': df.dtypes
    })
    #print("\nMissing percentages and data types for remaining columns:")
    #print(column_info)

    
    # Fill columns with a single unique value and NaN with 0 #
    unique_value_columns = []
    for col in df.columns:
        unique_values = df[col].dropna().unique()
        if len(unique_values) == 1 and df[col].isnull().any():
            unique_value = unique_values[0]
            unique_value_columns.append((col, unique_value))
    
    for col, unique_value in unique_value_columns:
        df[col] = df[col].fillna(0)
        test_df[col] = test_df[col].fillna(0)
    
    #print("Filled missing values in columns with a single unique value with 0.")
    

    #Categorising the features to apply imputers
    numeric_features, categorical_features, binary_features, features_to_drop = categorize_features(df)

    #print("Numeric features:", len(numeric_features))
    #print("Categorical features:", len(categorical_features))
    #print("Binary features:", len(binary_features))
    #print("Features to drop:", len(features_to_drop))

    #Handling missing values and scaling numeric features
    #Note: Changes made in train data-set are also meant to be done in test data-set
    for col in numeric_features:
        if col in df.columns:
            # Calculate median from training data
            median_val = df[col].median()
            # Fill missing values in both training and test
            df[col] = df[col].fillna(median_val)
            test_df[col] = test_df[col].fillna(median_val)
            
            # Scale the features
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val != 0:
                df[col] = (df[col] - mean_val) / std_val
                test_df[col] = (test_df[col] - mean_val) / std_val

    #Encoding categorical values
    for col in categorical_features:
        if col in df.columns:
            # Fill missing values with mode from training data
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            test_df[col] = test_df[col].fillna(mode_val)
            # One-hot encoding
            dummies_train = pd.get_dummies(df[col], prefix=col, drop_first=True)
            dummies_test = pd.get_dummies(test_df[col], prefix=col, drop_first=True)
            
            # Ensure test has all columns from training
            for col_dummy in dummies_train.columns:
                if col_dummy not in dummies_test.columns:
                    dummies_test[col_dummy] = 0
                    
            # Add encoded columns
            df = pd.concat([df, dummies_train], axis=1)
            test_df = pd.concat([test_df, dummies_test], axis=1)
            
            # Drop original categorical column
            df = df.drop(col, axis=1)
            test_df = test_df.drop(col, axis=1)
            
    #Handling binary features 
    for col in binary_features:
        if col in df.columns:
            # Fill missing values with mode from training data
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            test_df[col] = test_df[col].fillna(mode_val)
            
            # Convert to numeric if not already
            df[col] = pd.to_numeric(df[col], errors='coerce')
            test_df[col] = pd.to_numeric(test_df[col], errors='coerce')

    # Drop unnecessary features from the dataset
    for col in features_to_drop:
        if col in df.columns:
            df = df.drop(col, axis=1)
        if col in test_df.columns:
            test_df = test_df.drop(col, axis=1)
            
    test_df = test_df[df.columns]
    
    return df, test_df


# In[4]:


#Main function to handle data loading, preprocessing, training and prediction. 

def my_train_fn(*args, **kwargs):
    train = pd.read_csv("./train.csv")
    train = train.dropna(subset=['Target'])
    Y_train = train['Target']
    X_train = train.drop(columns=['Target'])
    
    test_file = kwargs.get("test_file", None)
    test_df = pd.read_csv(test_file) if test_file else None
    
    # Preprocess training and test data
    X_train, X_test = preprocess_data(X_train, test_df)
    
    label_map = {'low' : 0, 'medium' : 1, 'high' : 2}
    train['Target'] = train['Target'].map(label_map)
    
    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(Y_train), y=Y_train)
    class_weight_dict = {label: weight for label, weight in zip(np.unique(Y_train), class_weights)}
    weights = np.array([class_weight_dict[label] for label in Y_train])
    
    # Train model
    model = XGBClassifier(
        learning_rate=0.01,
        n_estimators=500,
        max_depth=8,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softmax',
        random_state=42
    )
    model.fit(X_train, Y_train, sample_weight=weights)
    
    return model, X_test   

def make_predictions(model, X_test ,test_fname, predictions_fname):
    
    preds = model.predict(X_test)
    
    label_map = {0: 'low',1 : 'medium',2 : 'high'}
    prediction_labels = [label_map[pred] for pred in preds]

    results_df = pd.DataFrame({
        'UID': test_fname['UID'],  # Ensure UID was preserved during preprocessing
        'Target': prediction_labels
    })
    results_df.to_csv(predictions_fname, index=False)




if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", type=str, help='file path of test.csv')
    parser.add_argument("--predictions-file", type=str, help='save path of predictions')
    args = parser.parse_args()
    #perform training
    #evaluation
    model, X_test = my_train_fn(test_file=args.test_file)
    make_predictions(model, X_test, args.test_file, args.predictions_file)



# In[ ]:




