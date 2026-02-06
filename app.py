# import pickle
# import numpy as np
# import os
# from flask import Flask, request, render_template, redirect, url_for

# app = Flask(__name__)

# # Paths
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BASE_DIR, 'Model', 'modelForPrediction.pkl')
# SCALER_PATH = os.path.join(BASE_DIR, 'Model', 'standardScaler.pkl')

# # Load model and scaler
# with open(MODEL_PATH, 'rb') as f:
#     model = pickle.load(f)

# with open(SCALER_PATH, 'rb') as f:
#     scaler = pickle.load(f)



# # Welcome page
# @app.route('/')
# def index():
#     return render_template('index.html')

# # Home page (form + result)
# @app.route('/home', methods=['GET', 'POST'])
# def home():
#     result = None

#     if request.method == 'POST':
#         try:
#             data = [
#                 float(request.form['Pregnancies']),
#                 float(request.form['Glucose']),
#                 float(request.form['BloodPressure']),
#                 float(request.form['SkinThickness']),
#                 float(request.form['Insulin']),
#                 float(request.form['BMI']),
#                 float(request.form['DiabetesPedigreeFunction']),
#                 float(request.form['Age'])
#             ]

#             scaled_data = scaler.transform([data])
#             prediction = model.predict(scaled_data)[0]

#             result = "🩺 Diabetic" if prediction == 1 else "✅ Not Diabetic"

#         except Exception as e:
#             result = f"Error: {str(e)}"

#     return render_template('home.html', result=result)

# # Run app
# if __name__ == "__main__":
#     app.run(debug=True)


import pickle
import numpy as np
import os
from flask import Flask, request, render_template

app = Flask(__name__)

# -------------------------
# Paths
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, 'Model', 'modelForPrediction.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'Model', 'standardScaler.pkl')

# -------------------------
# Load model and scaler
# -------------------------
with open(MODEL_PATH, 'rb') as model_file:
    model = pickle.load(model_file)

with open(SCALER_PATH, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# -------------------------
# Routes
# -------------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/home', methods=['GET', 'POST'])
def home():
    result = None

    if request.method == 'POST':
        try:
            data = [
                float(request.form['Pregnancies']),
                float(request.form['Glucose']),
                float(request.form['BloodPressure']),
                float(request.form['SkinThickness']),
                float(request.form['Insulin']),
                float(request.form['BMI']),
                float(request.form['DiabetesPedigreeFunction']),
                float(request.form['Age'])
            ]

            # Scale input
            scaled_data = scaler.transform([data])

            # Prediction
            prediction = model.predict(scaled_data)[0]

            if prediction == 1:
                result = "🩺 Diabetic"
            else:
                result = "✅ Not Diabetic"

        except Exception as e:
            result = f"Error: {e}"

    return render_template('home.html', result=result)

# -------------------------
# Run App (Render compatible)
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
