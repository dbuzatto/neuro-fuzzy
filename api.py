from fastapi import FastAPI
from pydantic import BaseModel

from inference import CLASSES, InferencePipeline

app = FastAPI(title="Neuro-Fuzzy Symptom Triage API", version="3.0")
pipeline = InferencePipeline()


class Symptoms(BaseModel):
    Fever: int
    Cough: int
    Fatigue: int
    DifficultyBreathing: int
    Age: int
    Gender: int
    BloodPressure: int
    CholesterolLevel: int


@app.get("/health")
def healthcheck():
    return {"status": "ok"}


@app.post("/predict")
def predict_symptom_severity(symptoms: Symptoms):
    label, probs = pipeline.predict({
        "Fever": symptoms.Fever,
        "Cough": symptoms.Cough,
        "Fatigue": symptoms.Fatigue,
        "Difficulty Breathing": symptoms.DifficultyBreathing,
        "Age": symptoms.Age,
        "Gender": symptoms.Gender,
        "Blood Pressure": symptoms.BloodPressure,
        "Cholesterol Level": symptoms.CholesterolLevel,
    })

    return {
        "predicted_class": label,
        "probabilities": {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))},
    }