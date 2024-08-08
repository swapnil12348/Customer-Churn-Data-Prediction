# Telco Customer Churn Prediction

## Project Description
This project is focused on building a machine learning model to predict customer churn for a telecommunications company. The key components of the project include:

- **Data Loading:** Loading the customer churn dataset from a CSV file.
- **Data Preprocessing:** Cleaning and preprocessing the data, handling missing values, encoding categorical variables, and preparing features and the target variable.
- **Exploratory Data Analysis (EDA):** Performing EDA to uncover insights and understand relationships between features and the target variable.
- **Model Training:** Training a Random Forest Classifier to predict customer churn.
- **Model Evaluation:** Assessing model performance using metrics like accuracy, classification reports, and confusion matrices.

## Features

- Data loading from a CSV file
- Data preprocessing and feature engineering
- Exploratory data analysis using Matplotlib and Seaborn
- Training and evaluating a Random Forest Classifier model
- Visualization of model performance

## Requirements

- Python 3.x
- pandas
- scikit-learn
- matplotlib
- seaborn

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/telco-customer-churn-prediction.git

2. Navigate to the project directory:

   ```bash
    cd telco-customer-churn-prediction
   
3. Create and activate a virtual environment (optional but recommended):
     ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`

4. Install the required packages:
    ```bash
    pip install -r requirements.txt

## Usage

1. Ensure that the dataset file WA_Fn-UseC_-Telco-Customer-Churn.csv is located in the data directory within the project.

2. Run the main script:

    ```bash
   python main.py

 This will execute the following steps:

 - Load the data from the CSV file
- Preprocess the data
- Perform exploratory data analysis
- Split the data into training and testing sets
- Train the Random Forest Classifier model
- Evaluate the model's performance
- The script will display the model's accuracy, classification report, and a confusion matrix visualization.

## Project Structure
- main.py: The main entry point of the project, which orchestrates the data loading, preprocessing, EDA, and model training.
- src/data_loader.py: Contains the function to load the data from the CSV file.
- src/data_preprocessor.py: Implements the data preprocessing and feature engineering logic.
- src/exploratory_analysis.py: Includes the functions to perform exploratory data analysis.
- src/model_trainer.py: Defines the function to train and evaluate the machine learning model.

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to check the issues page if you want to contribute.
