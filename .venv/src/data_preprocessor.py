import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preprocess_data(df):
    # dropping the customer id, this file is mostly concerned with data cleaning
    df = df.drop('customerID', axis=1)

    # Convert TotalCharges to numeric, replacing empty strings with NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Drop rows with NaN values
    df.dropna(inplace=True)

    # Encode categorical variables
    le = LabelEncoder()
    categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                        'PaperlessBilling', 'PaymentMethod', 'Churn']

    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    # Ensure all columns are numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop any rows with NaN values after conversion
    df.dropna(inplace=True)

    # Prepare features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    return X, y