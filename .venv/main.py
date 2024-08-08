from src.data_loader import load_data
from src.data_preprocessor import preprocess_data
from src.exploratory_analysis import perform_eda
from src.model_trainer import train_model
from sklearn.model_selection import train_test_split

def main():

    file_path = 'C:/Users/swapn/Customer Churn Prediction/.venv/data/WA_Fn-UseC_-Telco-Customer-Churn.csv'

    # Loading data
    df = load_data(file_path)

    # Preprocessing the data
    X, y = preprocess_data(df)

    # Performing exploratory data analysis
    perform_eda(X, y)

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training and evaluation of model
    model = train_model(X_train, y_train, X_test, y_test)

    print("Project completed successfully!")

if __name__ == "__main__":
    main()