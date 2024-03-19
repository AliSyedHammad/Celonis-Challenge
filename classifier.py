from fastapi import HTTPException
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier as SKRandomForestClassifier
from typing import List
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from abc import ABC
import joblib
import os
import numpy as np
from sklearn.svm import SVC
from sktime.classification.interval_based import TimeSeriesForestClassifier as SKTimeSeriesForestClassifier


from parsing_data import read_and_process_files


class TrainingInfo(BaseModel):
    classifier: str
    f1_score: float
    precision: float
    recall: float
    accuracy: float


class PredictionInfo(BaseModel):
    classifier: str
    classs: int

class Classifier(ABC):
    """
    Abstract base class for classifiers.
    """
    def __init__(self, test_size: float) -> None:
        self.test_size: float = test_size

    def __load_data(self):
        directory_path = 'files'
        data_dict = read_and_process_files(directory_path)
        return data_dict

    def extract_features_manualy(self):
        # Assume `gesture_data` is the dictionary returned by `read_and_process_files()`
        gesture_data = self.__load_data()  # Assuming this method is properly adjusted to be callable

        # Determine the total number of samples and the maximum length of any sample
        n_samples = sum(len(samples) for samples in gesture_data.values())
        n_features = max(max(len(sample) for sample in samples) for samples in gesture_data.values())

        # Initialize the data array and labels
        X = np.zeros((n_samples, 3 * n_features))
        y = np.zeros(n_samples)
        
        sample_index = 0
        for gesture_label, samples in gesture_data.items():
            for sample in samples:
                # Flatten the sample's (x, y, z) readings into a single feature array
                for i, (x, _y, z) in enumerate(sample):
                    X[sample_index, i] = x
                    X[sample_index, n_features + i] = _y
                    X[sample_index, 2 * n_features + i] = z
                # Set the gesture label for this sample
                y[sample_index] = gesture_label
                # Increment the sample index for the next iteration
                sample_index += 1

        # shuffling it
        p = np.random.permutation(n_samples)
        X = X[p, :]
        y = y[p]


        # Split the data into training 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=42)

        return X_train, X_test, y_train, y_test
    
    def train(self) -> TrainingInfo:
        """
        Train the classifier on the given training data.
        """
        pass

    
    def predict(self, file_content: str):
        """
        Make predictions using the trained classifier.
        """
        pass

    def format_data_for_prediction(self, file_content: str):
        """
        Format the data for prediction.
        """
        lines = file_content.strip().split('\n')
        data = np.array([list(map(float, line.split())) for line in lines if line.strip()])
        
        # Assume your model expects a single flattened array for a single prediction instance
        # Flattening the array
        data_flattened = data.flatten()
        
        # Reshape for a single sample if necessary, assuming your model was trained on multiple features
        # Example: if your model was trained on samples with 300 features
        num_features = self.X_train.shape[1]  # This assumes `self.X_train` reflects the training data's shape
        data_for_prediction = np.zeros((1, num_features))
        data_for_prediction[0, :data_flattened.shape[0]] = data_flattened
        
        return data_for_prediction

class LogisticRegressionClassifier(Classifier):
    def __init__(self, test_size: float):
        super().__init__(test_size=test_size)
        self.X_train, self.X_test, self.y_train, self.y_test = self.extract_features_manualy()

    def train(self):
        # Instantiate the random forest classifier
        self.model = LogisticRegression(max_iter=1000)

        self.model.fit(self.X_train, self.y_train)
        
        # check if saved_models folder exists 
        if not os.path.exists('saved_models'):
            os.makedirs('saved_models')
        # save this model using joblib
        joblib.dump(self.model, 'saved_models/logistic_regression_model.pkl')
        
        # Calculate evaluation metrics
        y_pred = self.model.predict(self.X_test)
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        accuracy = accuracy_score(self.y_test, y_pred)
        
        # Create TrainingInfo object
        training_info = TrainingInfo(
            classifier="LogisticRegressionClassifier",
            f1_score=f1,
            precision=precision,
            recall=recall,
            accuracy=accuracy
        )
        
        return training_info

    def predict(self, file_content):
        
        # check if there is a model file saved
        if os.path.exists('saved_models/logistic_regression_model.pkl'):
            self.model = joblib.load('saved_models/logistic_regression_model.pkl')
        else:
            raise HTTPException(status_code=400, detail="A training of the model must be called before prediction.")

        
        # Format the input data for prediction
        X_test = self.format_data_for_prediction(file_content)

        # Make predictions using the trained classifier
        prediction = self.model.predict(X_test)

        # Returning the prediction
        # Assuming your model predicts a class label
        return int(prediction[0])


class SVMClassifier(Classifier):
    def __init__(self, test_size: float):
        super().__init__(test_size=test_size)
        self.X_train, self.X_test, self.y_train, self.y_test = self.extract_features_manualy()

    def train(self):
        # Instantiate the random forest classifier
        self.model = SVC()

        self.model.fit(self.X_train, self.y_train)
        
        # check if saved_models folder exists 
        if not os.path.exists('saved_models'):
            os.makedirs('saved_models')
        # save this model using joblib
        joblib.dump(self.model, 'saved_models/svm_model.pkl')

        # Calculate evaluation metrics
        y_pred = self.model.predict(self.X_test)
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        accuracy = accuracy_score(self.y_test, y_pred)
        
        # Create TrainingInfo object
        training_info = TrainingInfo(
            classifier="RandomForestClassifier",
            f1_score=f1,
            precision=precision,
            recall=recall,
            accuracy=accuracy
        )
        
        return training_info

    def predict(self, file_content):
        
        # check if there is a model file saved
        if os.path.exists('saved_models/svm_model.pkl'):
            self.model = joblib.load('saved_models/svm_model.pkl')
        else:
            raise HTTPException(status_code=400, detail="A training of the model must be called before prediction.")

        # Format the input data for prediction
        X_test = self.format_data_for_prediction(file_content)
        
        # Make predictions using the trained classifier
        prediction = self.model.predict(X_test)

        # Returning the prediction
        # Assuming your model predicts a class label
        return int(prediction[0])