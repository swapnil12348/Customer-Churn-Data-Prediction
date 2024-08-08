import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(X, y):
    df = X.copy()
    df['Churn'] = y

    # Visualizing the distribution of numerical features
    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    for i, feature in enumerate(numerical_features):
        sns.histplot(df[feature], kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {feature}')
    plt.tight_layout()
    plt.show()

    # Visualizing the correlation between numerical features
    correlation_matrix = df[numerical_features].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix of Numerical Features')
    plt.show()

    # Visualizing churn rate
    churn_rate = df['Churn'].value_counts(normalize=True)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=churn_rate.index, y=churn_rate.values)
    plt.title('Churn Rate')
    plt.ylabel('Percentage')
    plt.xticks([0, 1], ['No Churn', 'Churn'])
    plt.show()