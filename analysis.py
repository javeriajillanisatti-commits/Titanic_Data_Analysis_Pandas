import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt

# Read the Titanic dataset
df = pd.read_csv("titanic.csv")

# Display first few rows
print("First 5 Rows:")
print(df.head())


# Check dataset details
print("\nDataset Information:")
print(df.info())


# Check missing values in dataset
print("\nMissing Values:")
print(df.isnull().sum())


# Data Cleaning

# Replace missing Age values with the average age
df["Age"] = df["Age"].fillna(df["Age"].mean())

# Replace missing Embarked values with the most common value
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Remove Cabin column because it contains many missing values
df.drop("Cabin", axis=1, inplace=True)


print("\nMissing Values After Cleaning:")
print(df.isnull().sum())


# Generate summary statistics
print("\nSummary Statistics:")
print(df.describe())


# Exploratory Data Analysis

# Count survived and not survived passengers
print("\nSurvival Count:")
print(df["Survived"].value_counts())


# Calculate survival percentage
print("\nSurvival Percentage:")
print(df["Survived"].value_counts(normalize=True) * 100)


# Analyze passenger genders
print("\nGender Distribution:")
print(df["Sex"].value_counts())


# Analyze passenger classes
print("\nPassenger Class Distribution:")
print(df["Pclass"].value_counts())


# Find average passenger age
print("\nAverage Age:")
print(df["Age"].mean())

# Survival Count Graph
plt.figure(figsize=(6,4))
df["Survived"].value_counts().plot(kind="bar")
plt.title("Passenger Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")


# Gender Distribution Graph
plt.figure(figsize=(6,4))
df["Sex"].value_counts().plot(kind="bar")
plt.title("Passenger Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")


# Passenger Class Distribution Graph
plt.figure(figsize=(6,4))
df["Pclass"].value_counts().sort_index().plot(kind="bar")
plt.title("Passenger Class Distribution")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")


# Show all graphs
plt.savefig("survival_graph.png")
plt.show()