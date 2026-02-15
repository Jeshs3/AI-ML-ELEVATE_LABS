import matplotlib
import pandas as pd

#Load the dataset
df=pd.read_csv("/content/drive/MyDrive/Elevate Labs/Titanic-Dataset.csv")

# Basic info
print("Shape:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())
print("\nBasic Statistics:\n", df.describe())

df.info()

df.tail(10)

"""Handle Missing values using Mean

"""

#Missing values are age, cabin, and embarked
df["Age"] = df["Age"].fillna(df["Age"].median())

#Cabin and Embarked are types object
df[["Cabin", "Embarked"]] = df[["Cabin", "Embarked"]].fillna("Unknown")

df.isnull().sum()

"""Convert Categorical Features"""

# pip install category_encoders

from sklearn.preprocessing import LabelEncoder
import category_encoders as ce

# Label Encoding
label_cols = ["Sex", "Cabin"]
le = LabelEncoder()
for col in label_cols:
    df[col] = le.fit_transform(df[col])

# Target Encoding for Embarked
target = df["Survived"]
encoder = ce.TargetEncoder(cols=["Embarked"])
df = encoder.fit_transform(df, target)

df.head()

"""Standardization"""

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[["Age", "Fare"]] = scaler.fit_transform(df[["Age", "Fare"]])

df.head(50)

"""Visualize outliers"""

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 3))
sns.boxplot(x=df["Fare"])
plt.title("Boxplot of Fare")
plt.show()

# Compute Q1, Q3, and IQR
Q1 = df["Fare"].quantile(0.25)
Q3 = df["Fare"].quantile(0.75)
IQR = Q3 - Q1

# Define bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers
df = df[(df["Fare"] >= lower_bound) & (df["Fare"] <= upper_bound)]

print("New shape after removing outliers:", df.shape)
