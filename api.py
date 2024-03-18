from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier as SKRandomForestClassifier
from typing import List

from model_service import ModelService

app = FastAPI()

@app.post("/api/train/{classifier_name}")
async def train_classifier(classifier_name: str, test_set_size: float):

    # Train the classifier and get training info
    training_info = ModelService.train_classifier(classifier_name, test_set_size)
    return training_info

@app.post("/api/predict/{classifier_name}")
async def predict_gesture(classifier_name: str, movement_data: str) -> List[int]:
    # Make predictions using the trained classifier
    predictions = ModelService.predict_gesture(classifier_name, movement_data)
    return predictions
