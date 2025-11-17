import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
import joblib


from main import ANFISLayer

model = tf.keras.models.load_model(
    "anfis_symptom_triage_model.keras",
    custom_objects={"ANFISLayer": ANFISLayer},
    compile=False
)

scaler = joblib.load("scaler.pkl")

app = FastAPI(title="Neuro-Fuzzy Symptom Triage API")

classes = ["Autocuidado", "Consulta Médica", "Emergência"]

class Symptoms(BaseModel):
    Fever: int
    Cough: int
    Fatigue: int
    DifficultyBreathing: int
    Age: int
    Gender: int
    BloodPressure: int
    CholesterolLevel: int

@app.post("/predict")
def predict_symptom_severity(symptoms: Symptoms):
    features = np.array([[
        symptoms.Fever,
        symptoms.Cough,
        symptoms.Fatigue,
        symptoms.DifficultyBreathing,
        symptoms.Age,
        symptoms.Gender,
        symptoms.BloodPressure,
        symptoms.CholesterolLevel
    ]])

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    predicted_class = int(np.argmax(prediction))

    return {
        "predicted_class": classes[predicted_class],
        "probabilities": {
            classes[i]: float(prediction[i]) for i in range(3)
        }
    }
