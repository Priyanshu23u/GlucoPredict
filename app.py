from flask import Flask, render_template, request, jsonify
import numpy as np
import os
import tensorflow as tf

# Load the trained model
model_path = "model.h5"
if not os.path.exists(model_path):
    from sklearn.linear_model import LogisticRegression
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    # Load dataset
    df = pd.read_csv("diabetes.csv")
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Save model
    model.save(model_path)
else:
    model = tf.keras.models.load_model(model_path)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = np.array([[float(request.form[key]) for key in request.form.keys()]])
        prediction = model.predict(data)
        result = "Diabetic" if prediction[0] > 0.5 else "Not Diabetic"
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

# HTML Frontend
html_content = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Diabetes Prediction</title>
    <style>
        body {
            background-color: #121212;
            color: white;
            font-family: Arial, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100vh;
        }
        .container {
            background-color: #1e1e1e;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(255, 255, 255, 0.1);
            width: 50%;
            text-align: center;
        }
        h1 {
            color: #ffcc00;
        }
        input, button {
            display: block;
            width: 100%;
            margin: 10px 0;
            padding: 10px;
            border-radius: 5px;
            border: none;
        }
        button {
            background-color: #6200ea;
            color: white;
            cursor: pointer;
        }
        footer {
            margin-top: 20px;
            color: #bbb;
        }
    </style>
</head>
<body>
    <h1>Welcome to the Diabetes Prediction System</h1>
    <div class="container">
        <h2>Enter Your Details</h2>
        <form id="diabetes-form">
            <input type="number" name="Pregnancies" placeholder="Pregnancies" min="0" max="20" required>
            <input type="number" name="Glucose" placeholder="Glucose" min="0" max="300" required>
            <input type="number" name="BloodPressure" placeholder="Blood Pressure" min="0" max="250" required>
            <input type="number" name="SkinThickness" placeholder="Skin Thickness" min="0" max="100" required>
            <input type="number" name="Insulin" placeholder="Insulin" min="0" max="1000" required>
            <input type="number" name="BMI" placeholder="BMI" min="0" max="100" step="0.1" required>
            <input type="number" name="DiabetesPedigreeFunction" placeholder="Diabetes Pedigree Function" min="0" max="3" step="0.01" required>
            <input type="number" name="Age" placeholder="Age" min="0" max="120" required>
            <button type="submit">Predict</button>
        </form>
        <h3 id="result"></h3>
    </div>
    <footer>© 2025 Diabetes Prediction System | All Rights Reserved</footer>
    <script>
        document.getElementById('diabetes-form').addEventListener('submit', function(event) {
            event.preventDefault();
            let formData = new FormData(this);
            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('result').innerText = data.prediction || data.error;
            });
        });
    </script>
</body>
</html>
'''

with open("templates/index.html", "w") as f:
    f.write(html_content)
