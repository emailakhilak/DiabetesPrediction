from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction route
@app.route("/predict", methods=["POST"])
def predict():

    age = float(request.form["Age"])
    glucose = float(request.form["Glucose"])
    bmi = float(request.form["BMI"])

    data = np.array([
        [age, glucose, bmi]
    ])

    # Apdirply same scaling used during training
    data = scaler.transform(data)

    prediction = model.predict(data)

    if prediction[0] == 1:
        result = "Diabetes Detected"
    else:
        result = "No Diabetes"

    return render_template(
        "result.html",
        result=result
    )


if __name__ == "__main__":
    app.run(debug=True)