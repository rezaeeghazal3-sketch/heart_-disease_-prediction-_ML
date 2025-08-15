# results_analysis.py
import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix

# Check what files exist in the data directory
print("=== FILES IN DATA DIRECTORY ===")
data_dir = 'C:/Users/Ghazal/Desktop/heart-disease-prediction/data/'
if os.path.exists(data_dir):
    files = os.listdir(data_dir)
    print("Available files:")
    for file in files:
        print(f"  - {file}")
else:
    print("Data directory not found!")

# Try different possible file names
possible_names = [
    'heart-disease-prediction.csv',
    'heart.csv',
    'heart_disease.csv',
    'heart_disease_prediction.csv',
    'dataset.csv',
    'data.csv'
]

df = None
for name in possible_names:
    file_path = f'C:/Users/Ghazal/Desktop/heart-disease-prediction/data/{name}'
    if os.path.exists(file_path):
        print(f"\nFound file: {name}")
        try:
            df = pd.read_csv(file_path)
            print(f"Successfully loaded: {name}")
            break
        except Exception as e:
            print(f"Error loading {name}: {e}")
            continue

if df is not None:
    print("\n=== DATASET OVERVIEW ===")
    print(f"Dataset shape: {df.shape}")
    print(f"Dataset columns: {df.columns.tolist()}")

    # Descriptive statistics for continuous variables
    print("\n=== DESCRIPTIVE STATISTICS ===")
    print("Continuous Variables:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        print(f"\n{col} statistics:")
        print(df[col].describe())

    # Frequency distribution for categorical variables
    print("\n=== CATEGORICAL VARIABLES FREQUENCY ===")
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    for col in df.columns:
        if df[col].dtype == 'object' or col in ['HeartDisease', 'ExerciseAngina']:
            print(f"\n{col}:")
            print("Counts:")
            print(df[col].value_counts())
            print("Percentages:")
            print((df[col].value_counts(normalize=True) * 100).round(2))

    # Check for missing values
    print("\n=== MISSING VALUES ===")
    print(df.isnull().sum())

    # Basic correlations if possible
    print("\n=== BASIC CORRELATIONS ===")
    if len(numeric_cols) > 1:
        print("Correlation matrix for numeric variables:")
        print(df[numeric_cols].corr().round(3))

    print("\n=== DATA TYPES ===")
    print(df.dtypes)

    print("\n=== FIRST 5 ROWS ===")
    print(df.head())
else:
    print("\nNo CSV file found! Please check the file name and location.")