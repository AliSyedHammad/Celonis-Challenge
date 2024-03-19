from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier as SKRandomForestClassifier
from typing import List
from fastapi.responses import JSONResponse

from model_service import ModelService

app = FastAPI()

@app.post("/api/train/{classifier_name}")
async def train_classifier(classifier_name: str, test_set_size: float):

    # Train the classifier and get training info
    training_info = ModelService.train_classifier(classifier_name, test_set_size)
    return training_info

# @app.post("/api/predict/{classifier_name}")
# async def predict_gesture(classifier_name: str, movement_data: str) -> List[int]:
#     # Make predictions using the trained classifier
#     predictions = ModelService.predict_gesture(classifier_name, movement_data)
#     return predictions


@app.post("/api/predict/{classifier_name}")
async def predict_gesture(classifier_name: str, file: UploadFile = File(...)):
    if file.content_type != 'text/plain':
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a text file.")

    try:
        content = await file.read()
        # Process or save your content here
        # For example, converting content to a string:
        text_content = content.decode('utf-8')
        # Assuming you're saving or processing the text content somehow
        # Now, return a response or the content back to the user
        
        predictions = ModelService.predict_gesture(classifier_name, text_content)
        return predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))