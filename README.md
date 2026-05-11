# CreditWise Loan System

CreditWise Loan System is a Machine Learning project that predicts whether a loan application is likely to be approved or rejected based on applicant financial and demographic information.

This project helps understand the complete ML workflow including:
- Data preprocessing
- Missing value handling
- Exploratory Data Analysis (EDA)
- Feature engineering
- Model training and evaluation

---

## Project Objective

The goal of this project is to build a loan approval prediction model that can classify applicants into:

- **Loan Approved**
- **Loan Not Approved**

using applicant information such as income, credit score, debt ratio, employment status, education, and property details.

---

## Dataset Features

The dataset contains features like:

- Applicant_ID
- Gender
- Marital_Status
- Education_Level
- Employment_Status
- Employer_Category
- Applicant_Income
- Coapplicant_Income
- Savings
- Credit_Score
- DTI_Ratio
- Loan_Purpose
- Property_Area
- Loan_Approved (Target Variable)

---

## Workflow

### 1. Data Loading
- Loaded dataset using Pandas

```python
df = pd.read_csv("loan_approval_data.csv")
```

---

### 2. Data Preprocessing
Performed preprocessing steps:

- Checked null values
- Handled missing numerical values using **mean imputation**
- Handled missing categorical values using **most frequent imputation**

Libraries used:
- `SimpleImputer`

---

### 3. Exploratory Data Analysis (EDA)

Visualizations created:

- Loan approval ratio (Pie chart)
- Gender distribution
- Applicant income histogram
- Co-applicant income histogram
- Boxplots for:
  - Applicant Income
  - Savings
  - DTI Ratio
  - Credit Score
- Credit score vs Loan approval
- Correlation heatmap

Libraries used:
- Matplotlib
- Seaborn

---

### 4. Feature Engineering

Applied transformations:

- Label Encoding
- One Hot Encoding
- Standard Scaling

Additional engineered features:
- `DTI_Ratio_sqr`
- `Credit_Score_sqr`
- `Applicant_Income_log`

These features improved model learning performance.

---

### 5. Train Test Split

Dataset split:

- Training: 80%
- Testing: 20%

```python
train_test_split(test_size=0.2, random_state=42)
```

---

## Models Used

Three machine learning models were trained:

### 1. Logistic Regression
Used for binary classification.

### 2. Gaussian Naive Bayes
Probabilistic classifier.

### 3. K-Nearest Neighbors (KNN)
Distance-based classification model.

---

## Evaluation Metrics

Models were evaluated using:

- Accuracy Score
- Precision Score
- Recall Score
- F1 Score
- Confusion Matrix

Example:

```python
accuracy_score()
precision_score()
recall_score()
f1_score()
confusion_matrix()
```

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

## Project Structure

```bash
CreditWiseLoanSystem/
│
├── CreditWiseLoanSystem.ipynb
├── loan_approval_data.csv
└── README.md
```

---

## Installation

Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

Run notebook:

```bash
jupyter notebook
```

---

## Future Improvements

- Hyperparameter tuning
- Random Forest / XGBoost models
- Streamlit deployment
- Model serialization using Pickle/Joblib
- Real-time loan prediction web app

---

## Learning Outcomes

From this project, I learned:

- End-to-end ML pipeline
- Data cleaning
- EDA visualization
- Feature engineering
- Classification models
- Model evaluation and optimization

---

## Author

**Aryan Lunagariya**

AI/ML Learning Project  
CreditWise Loan Prediction System
