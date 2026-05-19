# Diabetes Prediction Web Application using Logistic Regression

# Live Demo

Deployed Application:

https://diabetes-prediction-app-t6bc.onrender.com

---

# Project Overview

This project predicts whether a person has diabetes using Machine Learning and displays the prediction through a Flask web application.

The project uses:

- Python
- Pandas
- NumPy
- Scikit-Learn
- Logistic Regression
- Flask
- HTML
- CSS

Input features used:

- Age
- Glucose
- BMI

Output:

```text
0 → No Diabetes

1 → Diabetes
```

---

# Project Workflow

```text
Dataset
↓
Load Data
↓
Data Cleaning
↓
Handle Missing Values
↓
Feature Selection
↓
Split Data
↓
Balance Classes
↓
Scale Features
↓
Train Logistic Regression
↓
Evaluate Model
↓
Save Model
↓
Build Flask Website
↓
Take User Input
↓
Predict Diabetes
↓
Deploy on Render
```

---

# Project Folder Structure

```text
DiabetesPrediction/
│
├── dataset.csv
├── train_model.ipynb
├── train_model.py
├── model.pkl
├── scaler.pkl
├── app.py
├── requirements.txt
├── Procfile
├── README.md
│
├── templates/
│   ├── index.html
│   └── result.html
│
└── static/
    └── style.css
```

---

# Description of Files

| File | Purpose |
|--------|----------|
| dataset.csv | Diabetes dataset |
| train_model.ipynb | Learning and experimentation |
| train_model.py | Final training script |
| model.pkl | Saved trained model |
| scaler.pkl | Saved scaler |
| app.py | Flask backend |
| templates | HTML pages |
| static | CSS styling |
| requirements.txt | Dependency list |
| Procfile | Deployment configuration |
| README.md | Documentation |

---

# Installation

Clone repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Move into project folder:

```bash
cd DiabetesPrediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Model Information

Algorithm:

- Logistic Regression

Features:

- Age
- Glucose
- BMI

Preprocessing:

- Missing value handling
- Oversampling
- Feature scaling

Evaluation:

- Accuracy
- Confusion Matrix
- Classification Report

---

# Step 1: Load Dataset

```python
import pandas as pd

df = pd.read_csv("dataset.csv")
```

Check:

```python
print(df.shape)

print(df.info())

print(df.isnull().sum())
```

Purpose:

- Check rows
- Check columns
- Check data types
- Check missing values

---

# Step 2: Check Fake Missing Values

```python
print(
(df[["Glucose","BMI"]]==0).sum()
)
```

---

# Step 3: Replace Fake Missing Values

```python
df["Glucose"]=df["Glucose"].replace(
0,
df["Glucose"].median()
)

df["BMI"]=df["BMI"].replace(
0,
df["BMI"].median()
)
```

Reason:

Median handles unusual values better.

---

# Step 4: Select Features and Labels

```python
X=df[["Age","Glucose","BMI"]]

y=df["Outcome"]
```

---

# Step 5: Split Dataset

```python
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(
X,
y,
test_size=0.2,
random_state=42
)
```

---

# Step 6: Balance Training Data

```python
from imblearn.over_sampling import RandomOverSampler

oversample=RandomOverSampler(
random_state=42
)

X_train,y_train=oversample.fit_resample(
X_train,
y_train
)
```

---

# Step 7: Scale Features

```python
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

X_train=scaler.fit_transform(
X_train
)

X_test=scaler.transform(
X_test
)
```

---

# Step 8: Train Model

```python
from sklearn.linear_model import LogisticRegression

model=LogisticRegression()

model.fit(
X_train,
y_train
)
```

---

# Step 9: Check Accuracy

```python
accuracy=model.score(
X_test,
y_test
)

print(
"Accuracy:",
accuracy
)
```

---

# Step 10: Evaluate Model

Predictions:

```python
predictions=model.predict(
X_test
)
```

Confusion Matrix:

```python
from sklearn.metrics import confusion_matrix

print(
confusion_matrix(
y_test,
predictions
)
)
```

Classification Report:

```python
from sklearn.metrics import classification_report

print(
classification_report(
y_test,
predictions
)
)
```

---

# Step 11: Save Model

```python
import pickle

pickle.dump(
model,
open(
"model.pkl",
"wb"
)
)

pickle.dump(
scaler,
open(
"scaler.pkl",
"wb"
)
)
```

Generated:

```text
model.pkl

scaler.pkl
```

---

# Run Project Locally

Train model:

```bash
python train_model.py
```

Run Flask app:

```bash
python app.py
```

Open browser:

```text
http://127.0.0.1:5000
```

---

# GitHub and Render Deployment Guide

## Install Gunicorn

```bash
pip install gunicorn
```

---

## Create requirements.txt

```bash
pip freeze > requirements.txt
```

---

## Create Procfile

Create:

```text
Procfile
```

Add:

```text
web: gunicorn app:app
```

---

## Initialize Git

```bash
git init
```

---

## Add files

```bash
git add .
```

---

## Commit project

```bash
git commit -m "Initial Diabetes Prediction Project"
```

---

## Connect repository

```bash
git remote add origin YOUR_REPOSITORY_URL
```

---

## Push project

```bash
git push -u origin main
```

---

## Deploy on Render

Open:

https://render.com

Login with GitHub

Click:

```text
New +
↓
Web Service
```

Connect repository.

Set:

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
gunicorn app:app
```

Click:

```text
Deploy Web Service
```

---

# Updating Project

Whenever you modify code:

```bash
git add .

git commit -m "Updated project"

git push
```

Render automatically redeploys.

---

# Future Improvements

Possible improvements:

- Add more health parameters
- Add confidence score
- Improve UI
- Add charts
- Add database support
- Use advanced ML models

---

# Author

Akhila Kanneboina

AI/ML Enthusiast | Machine Learning Learner

GitHub: https://github.com/emailakhilak

LinkedIn: https://www.linkedin.com/in/kanneboina-akhila-76a1b1328/
