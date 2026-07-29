import pandas as pd

# Load dataset
df = pd.read_csv("dataset/retail_sales.csv")

print("========== DATASET OVERVIEW ==========")
print(df.head())

# Convert Date column to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Create new features
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Month Name"] = df["Date"].dt.month_name()
df["Day"] = df["Date"].dt.day
df["Day of Week"] = df["Date"].dt.day_name()

print("\n========== DATASET INFO ==========")
print(df.info())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATES ==========")
print(df.duplicated().sum())

print("\n========== NEW COLUMNS ==========")
print(df[["Date", "Year", "Month", "Month Name", "Day", "Day of Week"]].head())

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# -------------------------------
# 1. Gender Distribution
# -------------------------------
plt.figure(figsize=(6,5))
sns.countplot(x="Gender", data=df)
plt.title("Gender Distribution")
plt.savefig("images/gender_distribution.png")
plt.close()

# -------------------------------
# 2. Product Category Distribution
# -------------------------------
plt.figure(figsize=(7,5))
sns.countplot(x="Product Category", data=df)
plt.title("Product Category Distribution")
plt.savefig("images/product_category_distribution.png")
plt.close()

# -------------------------------
# 3. Age Distribution
# -------------------------------
plt.figure(figsize=(8,5))
plt.hist(df["Age"], bins=10)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.savefig("images/age_distribution.png")
plt.close()

# -------------------------------
# 4. Monthly Sales Trend
# -------------------------------
monthly_sales = df.groupby("Month Name")["Total Amount"].sum()

month_order = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]

monthly_sales = monthly_sales.reindex(month_order)

plt.figure(figsize=(10,5))
monthly_sales.plot(marker="o")
plt.title("Monthly Sales Trend")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("images/monthly_sales_trend.png")
plt.close()

# -------------------------------
# 5. Revenue by Category
# -------------------------------
plt.figure(figsize=(7,5))
sns.barplot(
    x=df.groupby("Product Category")["Total Amount"].sum().index,
    y=df.groupby("Product Category")["Total Amount"].sum().values
)
plt.title("Revenue by Product Category")
plt.savefig("images/revenue_by_category.png")
plt.close()

# -------------------------------
# 6. Quantity Sold by Category
# -------------------------------
plt.figure(figsize=(7,5))
sns.barplot(
    x=df.groupby("Product Category")["Quantity"].sum().index,
    y=df.groupby("Product Category")["Quantity"].sum().values
)
plt.title("Quantity Sold by Category")
plt.savefig("images/quantity_by_category.png")
plt.close()

# -------------------------------
# 7. Correlation Heatmap
# -------------------------------
plt.figure(figsize=(8,6))
sns.heatmap(
    df.select_dtypes(include="number").corr(),
    annot=True,
    cmap="Blues"
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("images/correlation_heatmap.png")
plt.close()

print("\nAll graphs have been saved successfully in the images folder!")

# -------------------------------
# Machine Learning - Product Category Prediction
# -------------------------------

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import joblib
import os

# Encode Gender
gender_encoder = LabelEncoder()
df["Gender"] = gender_encoder.fit_transform(df["Gender"])

# Encode Target (Product Category)
category_encoder = LabelEncoder()
df["Product Category"] = category_encoder.fit_transform(df["Product Category"])

# Features
X = df[[
    "Age",
    "Gender",
    "Quantity",
    "Price per Unit"
]]

# Target
y = df["Product Category"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL PERFORMANCE ==========")
print(f"Accuracy : {accuracy*100:.2f}%")

print("\nClassification Report")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

# Save Model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/product_category_model.pkl")

print("\nModel saved successfully!")