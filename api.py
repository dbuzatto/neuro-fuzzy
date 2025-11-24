from fastapi import FastAPI
from pydantic import BaseModel, Field

from inference import CLASSES, InferencePipeline

app = FastAPI(title="Neuro-Fuzzy Symptom Triage API", version="3.0")
pipeline = InferencePipeline()


class Symptoms(BaseModel):
    Fever: int = Field(..., description="1 = fever present, 0 = no fever", example=1)
    Cough: int = Field(..., description="1 = cough present, 0 = no cough", example=0)
    Fatigue: int = Field(..., description="1 = fatigue present, 0 = no fatigue", example=1)
    DifficultyBreathing: int = Field(..., description="1 = difficulty breathing, 0 = normal breathing", example=0)
    Age: int = Field(..., description="Patient age in years", example=35)
    Gender: int = Field(..., description="1 = Male, 0 = Female", example=1)
    BloodPressure: int = Field(..., description="0 = Low, 1 = Normal, 2 = High blood pressure", example=2)
    CholesterolLevel: int = Field(..., description="0 = Low, 1 = Normal, 2 = High cholesterol", example=1)


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