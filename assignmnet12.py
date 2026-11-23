
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Read Customer Churn Dataset
file_path = os.path.join(os.path.dirname(__file__), "Assig12inputfile.csv")
df = pd.read_csv(file_path)

print("First 5 Records:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nDataset Shape:")
print(df.shape)

# 2. Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# 3. Handle Missing Values
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

numeric_columns = df.select_dtypes(include=np.number).columns

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

categorical_columns = df.select_dtypes(include="object").columns

for column in categorical_columns:
    if df[column].isnull().sum() > 0:
        df[column] = df[column].fillna(df[column].mode()[0])

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# 4. Remove Duplicate Records
print("\nDuplicate Records:", df.duplicated().sum())
df = df.drop_duplicates()

# 5. Summary Statistics
print("\nSummary Statistics:")
print(df.describe())

# 6. Churn Analysis
print("\nCustomer Churn Count:")
print(df["Churn"].value_counts())

print("\nCustomer Churn Percentage:")
print(df["Churn"].value_counts(normalize=True) * 100)

# 7. NumPy Statistical Analysis
charges = df["MonthlyCharges"].to_numpy()

print("\nMonthly Charges Analysis:")
print("Mean:", np.mean(charges))
print("Median:", np.median(charges))
print("Minimum:", np.min(charges))
print("Maximum:", np.max(charges))
print("Standard Deviation:", np.std(charges))

# 8. Churn Distribution
df["Churn"].value_counts().plot(
    kind="bar",
    title="Customer Churn Distribution"
)
plt.xlabel("Churn Status")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

# 9. Monthly Charges Distribution
plt.hist(df["MonthlyCharges"], bins=20)
plt.title("Distribution of Monthly Charges")
plt.xlabel("Monthly Charges")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

# 10. Tenure vs Churn
df.groupby("Churn")["Tenure"].mean().plot(
    kind="bar",
    title="Average Tenure by Churn Status"
)
plt.xlabel("Churn Status")
plt.ylabel("Average Tenure")
plt.tight_layout()
plt.show()

# 11. Monthly Charges vs Churn
df.groupby("Churn")["MonthlyCharges"].mean().plot(
    kind="bar",
    title="Average Monthly Charges by Churn"
)
plt.xlabel("Churn Status")
plt.ylabel("Average Monthly Charges")
plt.tight_layout()
plt.show()

# 12. Correlation Analysis
numeric_data = df.select_dtypes(include=np.number)

print("\nCorrelation Matrix:")
print(numeric_data.corr())

print("\nEDA Completed Successfully!")
