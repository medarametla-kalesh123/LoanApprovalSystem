
# 🏦 Loan Approval Prediction System

A machine learning project that predicts whether a loan application will be approved or rejected based on applicant data.

## 📌 Overview

This project demonstrates an end-to-end machine learning workflow, including data preparation, feature engineering, model training, evaluation, and an interactive application for making predictions.

## ✨ Features

- 📊 Data analysis and preprocessing
- 🧹 Data cleaning and feature engineering
- 🤖 Loan approval classification using Logistic Regression
- 📈 Model evaluation with classification metrics
- 🖥️ Interactive application built with Streamlit
- 💾 Model artifact storage for reuse during prediction

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation |
| Scikit-learn | Model training and evaluation |
| Matplotlib | Data visualization |
| Seaborn | Exploratory data analysis |
| Streamlit | Interactive user interface |
| Joblib | Saving and loading model artifacts |

## 🔄 Machine Learning Workflow

1. Load the loan application dataset.
2. Inspect and preprocess the data.
3. Handle missing values and prepare features.
4. Split the data into training and testing sets.
5. Train the classification model.
6. Evaluate performance using suitable metrics.
7. Load the saved artifacts in the application.
8. Generate predictions for user-provided inputs.

## 📁 Project Structure

```text
LoanApprovalSystem/
├── data/
│   └── Loan_Data.csv
├── models/
├── outputs/
├── app.py
├── main.py
├── requirements.txt
└── README.md
```

## ⚙️ Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/medarametla-kalesh123/LoanApprovalSystem.git
cd LoanApprovalSystem
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

Review `main.py` to confirm that it performs model training, then run:

```bash
python main.py
```

### 5. Launch the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal.

## 📊 Model Evaluation

The model can be evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Add your actual evaluation results here after verifying them from a model run.

## 🎯 Learning Outcomes

- Practical machine learning workflow
- Data preprocessing and feature preparation
- Classification model development
- Model evaluation and artifact reuse
- Integrating machine learning into a Streamlit application

## 👨‍💻 Author

**Kaleswararao Medarametla**

- GitHub: [medarametla-kalesh123](https://github.com/medarametla-kalesh123)
- LinkedIn: [kaleswararao-medarametla](https://www.linkedin.com/in/kaleswararao-medarametla)
- Email: medarametlakalesh@gmail.com
