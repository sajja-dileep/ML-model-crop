import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score  # Using accuracy_score for classification
import pickle

# Load data
data = pd.read_csv("Crop_recommendation.csv")


print(f"Missing values:\n{data.isnull().sum()}")

# Handle missing values (if necessary)
# You can use techniques like filling with mean/median or removing rows

# Preprocess data
le = LabelEncoder()
data["label"] = le.fit_transform(data["label"])  # Encode categorical label


# Separate features (x) and target variable (y)
x = data.drop("label", axis=1)
y = data["label"]

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Train the model
model = SVC()
model.fit(x_train, y_train)

# Make predictions on the test set
y_pred = model.predict(x_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")  # Format accuracy to two decimal places

# Save the trained model (optional)
pickle.dump(model, open("model.pkl", "wb"))

# Crop recommendation function
def crop_recommend(temperature, humidity, rainfall, ph, N, P, K):
    names = np.array([[temperature, humidity, rainfall, ph, N, P, K]])
    prediction = model.predict(names).reshape(1, -1)
    crop = le.inverse_transform(prediction)[0]  # Decode prediction back to crop name
    return crop

# Get user input
N = int(input("Enter nitrogen content: "))
K = int(input("Enter potassium content: "))
P = int(input("Enter phosphorus content: "))
ph = float(input("Enter pH level: "))
rainfall =float(input("Enter rainfall (mm): "))
humidity = float(input("Enter humidity (%): "))
temperature = float(input("Enter temperature (°C): "))

# Predict crop and print result
predicted_crop = crop_recommend(temperature, humidity, rainfall, ph, N, P, K)
print(f"Suitable crop: {predicted_crop}")
