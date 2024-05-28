from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder

# Load the model (assuming "model.pkl" exists in the same directory)
try:
    model = pickle.load(open("model.pkl", "rb"))
except FileNotFoundError:
    print("Error: Model file not found!")
    exit()  # Handle the error appropriately

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("home.html")

@app.route("/predict", methods=['POST'])
def predict():
    try:

        N = int(request.form['Nitrogen'])
        P = int(request.form['Phosphorus'])
        K = int(request.form['Potassium'])
        temperature = float(request.form["Temperature"])
        ph = float(request.form['PH'])
        rainfall = float(request.form['Rainfall'])
        humidity = float(request.form['Humidity'])

        # Construct the feature vector
        features = [temperature, humidity, rainfall, ph, N, P, K]
        single_pred = np.array(features).reshape(1, -1)

        # Make prediction
        predict = model.predict(single_pred)

        # Mapping predicted label to crop name
        crop_names = {
            1: "Rice", 2: "Maize", 3: "Cotton", 4: "Coconut", 5: "Papaya", 6: "Orange",
            7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes",
            12: "Mango", 14: "Pomogranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean",
            13: "Banana", 18: "Mothbeans", 19: "Pigeonpeas", 21: "Chickpea", 20: "Kidneybeans",
            22: "Coffee"
        }

        crop = crop_names.get(predict[0], "No suitable crop found")
        mysol = "Suitable crop is {}".format(crop)

        return render_template('results.html', mysol=mysol)
    except ValueError:
        # Handle potential conversion errors from user input (e.g., non-numerical values)
        return render_template('home.html', error="Invalid input. Please enter numerical values.")




if __name__ == '__main__':
    app.run(debug=True)

