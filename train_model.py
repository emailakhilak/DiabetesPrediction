import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler

# Load dataset
df = pd.read_csv(
r"C:\Users\Andro\Desktop\AIML_Journey\Projects\DiabetesPrediction\dataset.csv"
)

# Replace suspicious zero values
df["Glucose"] = df["Glucose"].replace(
    0,
    df["Glucose"].median()
)

df["BMI"] = df["BMI"].replace(
    0,
    df["BMI"].median()
)

# Features and label
X = df[["Age", "Glucose", "BMI"]]
y = df["Outcome"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Balance training data only
oversample = RandomOverSampler(random_state=42)

X_train, y_train = oversample.fit_resample(
    X_train,
    y_train
)

# Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression()

model.fit(
    X_train,
    y_train
)

# Accuracy
accuracy = model.score(
    X_test,
    y_test
)

print("Accuracy:", accuracy)

# Save model and scaler
pickle.dump(
    model,
    open(
        r"C:\Users\Andro\Desktop\AIML_Journey\Projects\DiabetesPrediction\model.pkl",
        "wb"
    )
)

pickle.dump(
    scaler,
    open(
        r"C:\Users\Andro\Desktop\AIML_Journey\Projects\DiabetesPrediction\scaler.pkl",
        "wb"
    )
)
print("Model and scaler saved")

