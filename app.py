from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = pickle.load(open("health.pkl", "rb"))

@app.get("/")
def home():
    return {"message": "Health fitness weight prediction API is working with CI/CD"}

@app.post("/predict")
def predict(data: dict):
    exercise_minutes = float(data["exercise_minutes"])
    steps = float(data["steps"])
    food_calories = float(data["food_calories"])
    sleep_hours = float(data["sleep_hours"])
    water_intake_liters = float(data["water_intake_liters"])

    new_data = pd.DataFrame(
        [[exercise_minutes, steps, food_calories, sleep_hours, water_intake_liters]],
        columns=[
            "exercise_minutes",
            "steps",
            "food_calories",
            "sleep_hours",
            "water_intake_liters",
        ],
    )

    prediction = model.predict(new_data)

    return {
        "predicted_weight": float(prediction[0])
    }