import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# Create Required Folders
# ==========================================

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("data/Loan_Data.csv")

print("\nFirst 5 Rows")
print(df.head())

print("\nDataset Shape")
print(df.shape)

print("\nColumns")
print(df.columns)

print("\nDataset Information")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

print("\nStatistical Summary")
print(df.describe())

# ==========================================
# EDA
# ==========================================

# Loan Status Distribution
plt.figure(figsize=(6,4))
sns.countplot(x="Loan_Status", data=df)
plt.title("Loan Approval Distribution")
plt.savefig("outputs/loan_status_distribution.png")
plt.close()

# Gender Distribution
plt.figure(figsize=(6,4))
sns.countplot(x="Gender", data=df)
plt.title("Gender Distribution")
plt.savefig("outputs/gender_distribution.png")
plt.close()

# Education Distribution
plt.figure(figsize=(6,4))
sns.countplot(x="Education", data=df)
plt.title("Education Distribution")
plt.savefig("outputs/education_distribution.png")
plt.close()

# Property Area Distribution
plt.figure(figsize=(6,4))
sns.countplot(x="Property_Area", data=df)
plt.title("Property Area Distribution")
plt.savefig("outputs/property_area_distribution.png")
plt.close()

# Credit History vs Loan Status
plt.figure(figsize=(6,4))
sns.countplot(
    x="Credit_History",
    hue="Loan_Status",
    data=df
)

plt.title("Credit History vs Loan Approval")
plt.savefig(
    "outputs/credit_history_vs_loan_status.png"
)
plt.close()

# Applicant Income Distribution
plt.figure(figsize=(8,4))
sns.histplot(
    df["ApplicantIncome"],
    bins=30,
    kde=True
)

plt.title("Applicant Income Distribution")
plt.savefig(
    "outputs/applicant_income_distribution.png"
)
plt.close()

# Loan Amount Distribution
plt.figure(figsize=(8,4))
sns.histplot(
    df["LoanAmount"],
    bins=30,
    kde=True
)

plt.title("Loan Amount Distribution")
plt.savefig(
    "outputs/loan_amount_distribution.png"
)
plt.close()

# ==========================================
# Data Cleaning
# ==========================================

df["Gender"] = df["Gender"].fillna(
    df["Gender"].mode()[0]
)

df["Married"] = df["Married"].fillna(
    df["Married"].mode()[0]
)

df["Dependents"] = df["Dependents"].fillna(
    df["Dependents"].mode()[0]
)

df["Self_Employed"] = df["Self_Employed"].fillna(
    df["Self_Employed"].mode()[0]
)

df["Credit_History"] = df["Credit_History"].fillna(
    df["Credit_History"].mode()[0]
)

df["LoanAmount"] = df["LoanAmount"].fillna(
    df["LoanAmount"].mean()
)

df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(
    df["Loan_Amount_Term"].mean()
)

# ==========================================
# Encoding
# ==========================================

df["Gender"] = df["Gender"].map({
    "Male":1,
    "Female":0
})

df["Married"] = df["Married"].map({
    "Yes":1,
    "No":0
})

df["Education"] = df["Education"].map({
    "Graduate":1,
    "Not Graduate":0
})

df["Self_Employed"] = df["Self_Employed"].map({
    "Yes":1,
    "No":0
})

df["Property_Area"] = df["Property_Area"].map({
    "Rural":0,
    "Semiurban":1,
    "Urban":2
})

df["Loan_Status"] = df["Loan_Status"].map({
    "N":0,
    "Y":1
})

df["Dependents"] = df["Dependents"].replace(
    "3+",
    3
)

df["Dependents"] = df["Dependents"].astype(int)

# ==========================================
# Correlation Heatmap
# ==========================================

plt.figure(figsize=(10,8))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.tight_layout()

plt.savefig(
    "outputs/correlation_heatmap.png"
)

plt.close()

# ==========================================
# Feature Selection
# ==========================================

X = df.drop(
    ["Loan_ID", "Loan_Status"],
    axis=1
)

y = df["Loan_Status"]

# ==========================================
# Train Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================================
# Scaling
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================
# Model Training
# ==========================================

model = LogisticRegression(
    max_iter=5000
)

model.fit(
    X_train,
    y_train
)

# ==========================================
# Prediction
# ==========================================

predictions = model.predict(
    X_test
)

# ==========================================
# Evaluation
# ==========================================

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nAccuracy :", accuracy)

print("\nClassification Report")
print(
    classification_report(
        y_test,
        predictions
    )
)

# ==========================================
# Confusion Matrix
# ==========================================

cm = confusion_matrix(
    y_test,
    predictions
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig(
    "outputs/confusion_matrix.png"
)

plt.close()

# ==========================================
# Save Model
# ==========================================

joblib.dump(
    model,
    "models/loan_model.pkl"
)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

print("\nModel Saved Successfully")