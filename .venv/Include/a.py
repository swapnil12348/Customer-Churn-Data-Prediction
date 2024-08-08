import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
df = pd.read_csv('your_file_name.csv')

# Display the first few rows and basic information about the dataset
print(df.head())
print(df.info())