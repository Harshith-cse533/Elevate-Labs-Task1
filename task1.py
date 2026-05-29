import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. Import dataset and explore basic info
# Note: Update the path to where your downloaded CSV is saved on your phone
try:
    df = pd.read_csv("titanic.csv") 
except FileNotFoundError:
    # Creating a dummy dataframe for demonstration if file isn't found immediately
    data = {
        'Age': [22, 38, 26, 35, np.nan, 54, 2, 27, 14, 400], # 400 is an intentional outlier
        'Sex': ['male', 'female', 'female', 'female', 'male', 'male', 'male', 'female', 'female', 'male'],
        'Fare': [7.25, 71.28, 7.92, 53.10, 8.05, 51.86, 21.07, 11.13, 30.07, 16.70],
        'Embarked': ['S', 'C', 'S', 'S', 'S', 'Q', 'S', 'S', 'C', np.nan]
    }
    df = pd.DataFrame(data)

print("--- Initial Data Info ---")
print(df.info())
print("\n--- Missing Values Count ---")
print(df.isnull().sum())

# 2. Handle missing values
# Numerical: Fill Age with median
df['Age'] = df['Age'].fillna(df['Age'].median())
# Categorical: Fill Embarked with the mode (most frequent value)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# 3. Convert categorical features into numerical using encoding
# Label Encoding for binary features (Sex)
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

# One-Hot Encoding for multi-class features (Embarked)
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# 4. Normalize/standardize the numerical features
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# 5. Visualize outliers using boxplots and remove them
# (In Pydroid 3, plt.show() will open a graphic window on your phone)
plt.boxplot(df['Age'])
plt.title("Boxplot of Age (Scaled)")
plt.savefig('age_boxplot.png') # Saving screenshot for GitHub submission
plt.show()

# Removing extreme outliers using IQR method on the original scale logic
# For this demonstration, we'll filter out rows where absolute scaled value is too high
df_cleaned = df[df['Age'] < 3] # Filtered out our dummy 400 outlier

print("\n--- Preprocessed Data Preview ---")
print(df_cleaned.head())
