import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# load the dataset

file_path = "C:/puredata/statistical_methodology_dataset.csv"
data = pd.read_csv(file_path)

# display the first few rows of the dataset

print("Initial Dataset Preview:")
print(data.head())

# handle missing values

def handle_missing_values(df):
    """Handle missing values by filling numeric columns with the mean and categorical columns with the mode."""
    for column in df.columns:
        if df[column].dtype in ['int64', 'float64']:
            df[column].fillna(df[column].mean(), inplace=True)
        else:
            df[column].fillna(df[column].mode()[0], inplace=True)

handle_missing_values(data)

# encode categorical variables

def encode_categorical(df, categorical_columns):
    """Encode categorical columns using LabelEncoder."""
    encoder = LabelEncoder()
    for column in categorical_columns:
        df[column] = encoder.fit_transform(df[column])

categorical_columns = ['education_level', 'region']
encode_categorical(data, categorical_columns)

# normalize numeric columns

def normalize_numeric(df, numeric_columns):
    """Normalize numeric columns to the range [0, 1] using MinMaxScaler."""
    scaler = MinMaxScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

numeric_columns = ['age', 'income', 'hours_worked_per_week', 'job_satisfaction', 'mental_health_score', 'physical_health_score']
normalize_numeric(data, numeric_columns)

# save the preprocessed dataset

output_path = "C:/puredata/statistical_methodology_preprocessed.csv"
data.to_csv(output_path, index=False)
print(f"Preprocessed dataset saved to: {output_path}")

