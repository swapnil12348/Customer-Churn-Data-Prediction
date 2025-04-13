import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Page configuration
st.set_page_config(page_title="Customer Churn Prediction", page_icon=":bar_chart:", layout="wide")


def display_dataframe_info(df):
    """Display comprehensive dataframe information in Streamlit"""
    st.header("Data Overview")

    # Dataset basic information
    st.subheader("Dataset Dimensions")
    st.write(f"Number of Rows: {df.shape[0]}")
    st.write(f"Number of Columns: {df.shape[1]}")

    # Display first few rows
    st.subheader("First Few Rows")
    st.dataframe(df.head())

    # Column information
    st.subheader("Column Information")
    column_info = pd.DataFrame({
        'Column Name': df.columns,
        'Data Type': df.dtypes,
        'Non-Null Count': df.count(),
        'Null Count': df.isnull().sum(),
        'Null Percentage': (df.isnull().sum() / len(df) * 100).round(2)
    })
    st.dataframe(column_info)

    # Basic statistics for numeric columns
    st.subheader("Numeric Columns Statistics")
    st.dataframe(df.describe())


def load_data(uploaded_file=None):
    """Load data from file or sample dataset"""
    # List of potential file paths to try
    potential_paths = [
        'C:/Users/swapn/CustomerChurnPrediction/.venv/data/WA_Fn-UseC_-Telco-Customer-Churn.csv',
        'C:/Users/swapn/CustomerChurnPrediction/data/WA_Fn-UseC_-Telco-Customer-Churn.csv',
        './data/WA_Fn-UseC_-Telco-Customer-Churn.csv',
        '../data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    ]

    # If a file is uploaded through Streamlit, use that first
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("Successfully loaded uploaded file!")
            return df
        except Exception as e:
            st.error(f"Error loading uploaded file: {e}")

    # Try loading from potential local paths
    for path in potential_paths:
        try:
            df = pd.read_csv(path)
            st.success(f"Successfully loaded data from {path}")
            return df
        except FileNotFoundError:
            continue
        except Exception as e:
            st.error(f"Error loading file from {path}: {e}")

    # If no file is found, create a minimal sample dataset
    st.warning("No dataset found. Creating a minimal sample dataset.")
    df = pd.DataFrame({
        'customerID': ['1234', '5678'],
        'gender': ['Male', 'Female'],
        'SeniorCitizen': [0, 1],
        'Partner': ['Yes', 'No'],
        'Dependents': ['No', 'Yes'],
        'tenure': [10, 20],
        'PhoneService': ['Yes', 'No'],
        'MultipleLines': ['No', 'Yes'],
        'InternetService': ['DSL', 'Fiber optic'],
        'OnlineSecurity': ['No', 'Yes'],
        'OnlineBackup': ['Yes', 'No'],
        'DeviceProtection': ['No', 'Yes'],
        'TechSupport': ['Yes', 'No'],
        'StreamingTV': ['No', 'Yes'],
        'StreamingMovies': ['Yes', 'No'],
        'Contract': ['Month-to-month', 'One year'],
        'PaperlessBilling': ['Yes', 'No'],
        'PaymentMethod': ['Electronic check', 'Mailed check'],
        'MonthlyCharges': [29.85, 56.95],
        'TotalCharges': [29.85, 1139.50],
        'Churn': [0, 1]
    })
    return df


def preprocess_data(df):
    """Preprocess the data for model training"""
    # Drop customer ID
    df = df.drop('customerID', axis=1)

    # Convert TotalCharges to numeric
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

    return X, y, df


def train_model(X_train, y_train):
    """Train a Random Forest Classifier"""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model


def individual_churn_prediction(model, X):
    """
    Create interactive widgets for individual churn prediction

    Args:
        model: Trained Random Forest Classifier
        X: Feature DataFrame used for training (to get feature names and ranges)
    """
    st.subheader("Customer Churn Probability Predictor")

    # Create columns for better layout
    col1, col2 = st.columns(2)

    # Categorical Features
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    with col2:
        online_security = st.selectbox("Online Security", ["No", "Yes"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

    # Additional features
    col3, col4 = st.columns(2)

    with col3:
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox("Payment Method",
                                      ["Electronic check", "Mailed check", "Bank transfer (automatic)",
                                       "Credit card (automatic)"])

    with col4:
        tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
        monthly_charges = st.slider("Monthly Charges ($)", min_value=0.0, max_value=120.0, value=50.0, step=0.5)
        total_charges = st.slider("Total Charges ($)", min_value=0.0, max_value=8000.0, value=1000.0, step=50.0)

    # Prediction button
    if st.button("Predict Churn Probability"):
        # Prepare input data (matching the preprocessing steps)
        input_data = pd.DataFrame({
            'gender': [gender],
            'SeniorCitizen': [1 if senior_citizen == "Yes" else 0],
            'Partner': [partner],
            'Dependents': [dependents],
            'tenure': [tenure],
            'PhoneService': [phone_service],
            'MultipleLines': [multiple_lines],
            'InternetService': [internet_service],
            'OnlineSecurity': [online_security],
            'OnlineBackup': [online_backup],
            'DeviceProtection': [device_protection],
            'TechSupport': [tech_support],
            'StreamingTV': [streaming_tv],
            'StreamingMovies': [streaming_movies],
            'Contract': [contract],
            'PaperlessBilling': [paperless_billing],
            'PaymentMethod': [payment_method],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges]
        })

        # Use the preprocessing function from before
        le = LabelEncoder()
        categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                            'PaperlessBilling', 'PaymentMethod']

        for col in categorical_cols:
            input_data[col] = le.fit_transform(input_data[col])

        # Ensure all columns are numeric
        for col in input_data.columns:
            input_data[col] = pd.to_numeric(input_data[col], errors='coerce')

        # Predict churn probability
        prediction_probs = model.predict_proba(input_data)
        no_churn_prob, churn_prob = prediction_probs[0]


        # Display results
        st.subheader("Prediction Results")
        st.metric("Probability of Churning", f"{churn_prob * 100:.2f}%")
        st.metric("Probability of Staying", f"{no_churn_prob * 100:.2f}%")

        # Interpret the results
        if churn_prob > 0.5:
            st.warning("High risk of customer churn! Recommend retention strategies.")
        else:
            st.success("Low risk of customer churn. Customer seems satisfied.")


def main():
    st.title("🚀 Customer Churn Prediction Dashboard")

    # Sidebar for navigation
    menu = ["Data Overview", "Exploratory Data Analysis", "Model Training", "Churn Prediction"]
    choice = st.sidebar.selectbox("Select Module", menu)

    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload your Telco Customer Churn CSV", type=['csv'])

    # Load data
    df = load_data(uploaded_file)

    # Preprocess data
    X, y, full_df = preprocess_data(df)

    # Store the model as a session state variable to persist across reruns
    if 'model' not in st.session_state:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Train model
        st.session_state.model = train_model(X_train, y_train)

    if choice == "Data Overview":
        display_dataframe_info(df)

    elif choice == "Exploratory Data Analysis":
        st.header("Exploratory Data Analysis")

        # Numerical Features Distribution
        st.subheader("Numerical Features Distribution")
        numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i, feature in enumerate(numerical_features):
            sns.histplot(full_df[feature], kde=True, ax=axes[i])
            axes[i].set_title(f'Distribution of {feature}')
        plt.tight_layout()
        st.pyplot(fig)

        # Correlation Heatmap
        st.subheader("Correlation Heatmap")
        correlation_matrix = full_df[numerical_features].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        # Churn Rate
        st.subheader("Churn Rate")
        churn_rate = full_df['Churn'].value_counts(normalize=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x=churn_rate.index, y=churn_rate.values, ax=ax)
        ax.set_title('Churn Rate')
        ax.set_ylabel('Percentage')
        ax.set_xticklabels(['No Churn', 'Churn'])
        st.pyplot(fig)

    elif choice == "Model Training":
        st.header("Model Training")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = train_model(X_train, y_train)

        # Model evaluation
        y_pred = model.predict(X_test)

        st.subheader("Model Performance")
        st.write("Accuracy:", accuracy_score(y_test, y_pred))

        # Classification Report
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        st.pyplot(fig)

        # Feature Importance (add this section)
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance[:10], ax=ax)
        ax.set_title('Top 10 Feature Importances')
        st.pyplot(fig)

    elif choice == "Churn Prediction":
        # Call the new individual prediction function
        individual_churn_prediction(st.session_state.model, X)


if __name__ == "__main__":
    main()
