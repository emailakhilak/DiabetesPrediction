# Diabetes Prediction Web Application using Logistic Regression

## Project Overview

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
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── static/
│   └── style.css
│
└── README.md
```

---

# Description of Files

| File | Purpose |
|--------|---------|
| dataset.csv | Diabetes dataset |
| train_model.ipynb | Learning, experiments, testing |
| train_model.py | Final training script |
| model.pkl | Saved trained model |
| scaler.pkl | Saved scaler |
| app.py | Flask backend |
| templates | HTML pages |
| static | CSS styling |
| README.md | Project documentation |

---

# Step 1: Create Project Folder

Create:

```text
AIML_Journey/
│
└── Projects/
    │
    └── DiabetesPrediction/
```

Move dataset into this folder.

Example:

```text
DiabetesPrediction/
│
└── dataset.csv
```

---

# Step 2: Install Required Libraries

Open terminal:

```bash
pip install pandas
pip install numpy
pip install scikit-learn
pip install flask
pip install imbalanced-learn
```

Or:

```bash
pip install pandas numpy scikit-learn flask imbalanced-learn
```

Check installed packages:

```bash
pip list
```

---

# Step 3: Create Notebook

Create:

```text
train_model.ipynb
```

Purpose:

- Learning
- Experimentation
- Testing
- Evaluation

---

# Step 4: Load Dataset

```python
import pandas as pd

df=pd.read_csv("dataset.csv")
```

Check data:

```python
print(df.shape)

print(df.info())

print(df.isnull().sum())
```

Purpose:

- Check rows
- Check columns
- Check datatypes
- Check missing values

---

# Step 5: Check Fake Missing Values

Some datasets store missing values as:

```text
0
```

instead of:

```text
NaN
```

Check:

```python
print(
(df[["Glucose","BMI"]]==0).sum()
)
```

---

# Step 6: Replace Fake Missing Values

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

# Step 7: Select Features and Labels

Features:

```python
X=df[["Age","Glucose","BMI"]]
```

Label:

```python
y=df["Outcome"]
```

---

# Step 8: Split Dataset

```python
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(
X,
y,
test_size=0.2,
random_state=42
)
```

Meaning:

```text
80% → Training Data

20% → Testing Data
```

---

# Step 9: Balance Training Data

Check imbalance:

```python
print(y.value_counts())
```

Balance:

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

Important:

Balance only:

```text
X_train
y_train
```

Do NOT balance before splitting.

Reason:

```text
Avoid data leakage
```

---

# Step 10: Scale Features

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

Reason:

Feature ranges differ:

```text
Age → 50

BMI → 30

Glucose → 180
```

Scaling converts features into similar ranges.

---

# Step 11: Train Logistic Regression Model

```python
from sklearn.linear_model import LogisticRegression

model=LogisticRegression()

model.fit(
X_train,
y_train
)
```

Purpose:

```text
Train model using training data
```

---

# Step 12: Check Accuracy

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

# Step 13: Evaluate Model

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

# Step 14: Save Model and Scaler

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

Generated files:

```text
model.pkl

scaler.pkl
```

---

# Step 15: Create Final Training Script

Create:

```text
train_model.py
```

Move final cleaned code into this file.

Purpose:

- Final production code
- Generate model.pkl
- Generate scaler.pkl

---

# Step 16: Train Model Using Terminal

Move into project:

```bash
cd Projects\DiabetesPrediction
```

Run:

```bash
python train_model.py
```

Expected output:

```text
Accuracy: 0.80...

Model and scaler saved
```

After successful execution:

```text
DiabetesPrediction/
│
├── model.pkl
├── scaler.pkl
```

---

# Step 17: Create Flask Application

Create:

```text
app.py
```

Responsibilities:

- Load model
- Load scaler
- Accept user input
- Scale input
- Predict output
- Display result

Run:

```bash
python app.py
```

Expected:

```text
Running on:

http://127.0.0.1:5000
```

---

# Step 18: Create Frontend Files

Inside:

```text
templates/
```

Create:

```text
index.html

result.html
```

Inside:

```text
static/
```

Create:

```text
style.css
```

Purpose:

```text
index.html
→ User input page

result.html
→ Display prediction

style.css
→ Styling
```

---

Open browser:

```text
http://127.0.0.1:5000
```

---

# Future Improvements

Possible upgrades:

- Add more health parameters
- Add confidence score
- Improve UI
- Deploy online
- Add charts
- Add database
- Use advanced models

---

# Author

Akhila Kanneboina

AI/ML Learning Journey Project
