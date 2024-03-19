from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from classifier import SVMClassifier, TrainingInfo, LogisticRegressionClassifier

class TrainRequest(BaseModel):
    test_set_size: float
    classifier_name: str


class ModelService:
    @classmethod
    def train_classifier(cls, classifier_name: str, test_set_size: str) -> TrainingInfo:

        if classifier_name == "logistic_regression_classifier":
            classifier = LogisticRegressionClassifier(test_size=test_set_size)
        elif classifier_name == "svm_classifier":
            classifier  = SVMClassifier(test_size=test_set_size)
        else:
            raise HTTPException(status_code=400, detail=f"Classifier '{classifier_name}' not supported.")

        training_info: TrainingInfo = classifier.train()

        return training_info

    @classmethod
    def predict_gesture(cls, classifier_name: str, file_content: str) -> List[int]:
        if classifier_name == "logistic_regression_classifier":
            classifier = LogisticRegressionClassifier(test_size=0.1)
        elif classifier_name == "svm_classifier":
            classifier  = SVMClassifier(test_size=0.1)
        else:
            raise HTTPException(status_code=400, detail=f"Classifier '{classifier_name}' not supported.")

        predictions = classifier.predict(file_content)

        return predictions
