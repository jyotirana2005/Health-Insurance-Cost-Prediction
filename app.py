from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the pre-trained models and other necessary components
try:
    with open('model/linear_regression.pkl', 'rb') as f:
        model_lr = pickle.load(f)
    with open('model/svr.pkl', 'rb') as f:
        model_svr = pickle.load(f)
    with open('model/decision_tree.pkl', 'rb') as f:
        model_dt = pickle.load(f)
    with open('model/random_forest.pkl', 'rb') as f:
        model_rf = pickle.load(f)
    with open('model/scaler_X.pkl', 'rb') as f:
        scaler_X = pickle.load(f)
    with open('model/scaler_y.pkl', 'rb') as f:
        scaler_y = pickle.load(f)    
    with open('model/label_encoder_sex.pkl', 'rb') as f:
        le_sex = pickle.load(f)
    with open('model/label_encoder_smoker.pkl', 'rb') as f:
        le_smoker = pickle.load(f)
    
    print("All models and scalers loaded successfully.")
except Exception as e:
    print(f"Error loading models/scalers: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract and print form data
        age = float(request.form.get('age', 0))
        sex = request.form.get('sex', '')
        bmi = float(request.form.get('bmi', 0))
        children = int(request.form.get('children', 0))
        smoker = request.form.get('smoker', '')
        region = request.form.get('region', '')

        print(f"Extracted data: age={age}, sex={sex}, bmi={bmi}, children={children}, smoker={smoker}, region={region}")

        # Encode categorical features
        try:
            sex_encoded = le_sex.transform([sex])[0]
            smoker_encoded = le_smoker.transform([smoker])[0]
        except ValueError as e:
            print(f"Error in encoding categorical variables: {e}")
            return render_template('result.html', prediction_max="Error")

        # Prepare input data
        region_encoded = np.zeros(4)
        regions = ['northeast', 'northwest', 'southeast', 'southwest']
        if region in regions:
            region_encoded[regions.index(region)] = 1
        else:
            print(f"Invalid region value: {region}")
            return render_template('result.html', prediction_max="Error")

        input_data = np.array([[age, sex_encoded, bmi, children, smoker_encoded] + list(region_encoded)])
        input_data = scaler_X.transform(input_data)

        print(f"Processed input data: {input_data}")

        # Make predictions using different models
        prediction_lr = model_lr.predict(input_data)
        prediction_svr = model_svr.predict(input_data)
        prediction_dt = model_dt.predict(input_data)
        prediction_rf = model_rf.predict(input_data)

        print(f"Predictions: LR={prediction_lr}, SVR={prediction_svr}, DT={prediction_dt}, RF={prediction_rf}")

        # Inverse transform the predictions to original scale
        prediction_lr = scaler_y.inverse_transform(prediction_lr.reshape(-1, 1))[0][0]
        prediction_svr = scaler_y.inverse_transform(prediction_svr.reshape(-1, 1))[0][0]
        prediction_dt = scaler_y.inverse_transform(prediction_dt.reshape(-1, 1))[0][0]
        prediction_rf = scaler_y.inverse_transform(prediction_rf.reshape(-1, 1))[0][0]

        print(f"Inverse transformed predictions: LR={prediction_lr}, SVR={prediction_svr}, DT={prediction_dt}, RF={prediction_rf}")

        # Get the prediction from the most accurate model
        # For simplicity, assume we compare predictions based on model accuracy
        predictions = {
            'Linear Regression': prediction_lr,
            'SVR': prediction_svr,
            'Decision Tree': prediction_dt,
            'Random Forest': prediction_rf
        }
        
        # Here, we are assuming we know the model accuracies and choosing the best one
        best_model_name = 'Random Forest' 
        prediction_max = predictions[best_model_name]

        return render_template('result.html', prediction_max=str(prediction_max))

    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template('result.html', prediction_max="Error")

if __name__ == '__main__':
    app.run(debug=True)
